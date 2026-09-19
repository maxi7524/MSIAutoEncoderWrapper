"""Reproducible synthetic spectra with complete generator-owned ion labels."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from msi_dataset_manager.annotations.chemistry import parse_formula

from ..annotation_evidence import IonCatalogue
from ..batches import SpectrumBatch
from ..spaces import SpectrumSpace
from ..supervision_masks import simulated_negative_mask_key
from ..targets import TargetBatch, TargetSchema
from ...utils.logger import get_custom_logger
from .sampling import (
    SyntheticSamplingContext,
    SyntheticSamplingPlanEntry,
    get_sampling_strategy,
)
from .annotation_population import AnnotationPopulation
from .representations import (
    SyntheticRepresentationContext,
    SyntheticRepresentationSpec,
    get_representation_strategy,
)
from .sources import CandidateCatalogPeakSource, CataloguePeakSource, SyntheticPeakSource

logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class SyntheticSpectrumConfig:
    """Control the geometry and sampling of a synthetic pretraining population.

    :param samples: Number of generated spectra per epoch.
    :param seed: Local seed, independent of global NumPy and Torch state.
    :param max_peaks: Upper bound on sampled components per spectrum.
    :param modes: Legacy mixture modes; repeated names increase their sampling
        frequency when ``sampling_plan`` is not configured.
    :param sampling_plan: Exact per-epoch strategy counts.  Every entry has
        ``strategy``, ``count``, optional ``parameters``, and optional
        ``label_targets``.
    :param label_targets: Whether synthetic targets are available by default.
        Set to ``False`` for reconstruction-only synthetic samples.
    :param peak_radius: Half-width of a triangular peak in bins; zero is a point.
    :param normalization: ``tic``, ``max``, ``l2``, or ``none``.
    """

    samples: int = 4096
    seed: int = 42
    max_peaks: int = 16
    modes: tuple[str, ...] = ("single_random", "single_annotated", "random", "annotated", "mixed")
    sampling_plan: tuple[SyntheticSamplingPlanEntry | Mapping[str, Any], ...] = ()
    label_targets: bool = True
    peak_radius: int = 0
    normalization: str = "tic"
    representation: SyntheticRepresentationSpec | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in ("samples", "max_peaks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if isinstance(self.peak_radius, bool) or not isinstance(self.peak_radius, int) or self.peak_radius < 0:
            raise ValueError("peak_radius must be a nonnegative integer.")
        if not isinstance(self.label_targets, bool):
            raise ValueError("label_targets must be a boolean.")
        normalized_plan = tuple(SyntheticSamplingPlanEntry.from_value(entry) for entry in self.sampling_plan)
        object.__setattr__(self, "sampling_plan", normalized_plan)
        representation = SyntheticRepresentationSpec.from_value(self.representation)
        if representation.strategy == "triangular_peak" and "peak_radius" not in representation.parameters:
            representation = SyntheticRepresentationSpec(
                strategy=representation.strategy,
                parameters={**representation.parameters, "peak_radius": self.peak_radius},
            )
        get_representation_strategy(representation.strategy, **representation.parameters)
        for entry in normalized_plan:
            if entry.representation is None:
                continue
            entry_representation = SyntheticRepresentationSpec.from_value(entry.representation)
            get_representation_strategy(
                entry_representation.strategy,
                **entry_representation.parameters,
            )
        object.__setattr__(self, "representation", representation)
        if normalized_plan:
            if sum(entry.output_count for entry in normalized_plan) != self.samples:
                raise ValueError(
                    "sampling_plan output counts, including candidate permutations, "
                    "must sum exactly to samples."
                )
            for entry in normalized_plan:
                get_sampling_strategy(entry.strategy, **entry.parameters)
        elif not self.modes:
            raise ValueError("At least one legacy mode or sampling_plan entry is required.")
        else:
            for mode in self.modes:
                get_sampling_strategy(mode)
        if self.normalization not in {"tic", "max", "l2", "none"}:
            raise ValueError("Unsupported synthetic normalization.")


@dataclass(frozen=True)
class SyntheticSpectrumSample:
    """Tuple-compatible synthetic example with generator provenance.

    The four-item iteration contract preserves existing dataset consumers.
    ``metadata`` is consumed by :meth:`SyntheticSpectrumDataset.collate_fn`.
    """

    sample_id: int
    spectrum: torch.Tensor
    values: Mapping[str, torch.Tensor]
    masks: Mapping[str, torch.Tensor]
    metadata: Mapping[str, Any]

    def __iter__(self) -> Iterator[Any]:
        """Yield the historical four-item dataset sample contract."""
        return iter((self.sample_id, self.spectrum, self.values, self.masks))

    def __getitem__(self, index: int) -> Any:
        """Provide positional access compatible with the historic tuple sample."""
        return (self.sample_id, self.spectrum, self.values, self.masks)[index]


class SyntheticSpectrumDataset(Dataset):
    """Generate examples by index and epoch, without reading production spectra.

    :param catalogue: Legacy annotation-derived ion labels and coordinates.
    :param peak_source: Abstract labelled peak source.  This is the extension
        point for future candidate-derived coordinates.
    :param mass_axis: Global model axis, shape ``(M,)``.
    :param schemas: Shared real/synthetic target schemas.
    :param config: Generator configuration.
    :param eligible_ions: Optional training-only ion indices.
    :param chemistry: Optional frozen ion descriptions.
    :param dtype: Numerical dtype of generated spectra.

    REMARK: Missing ranges in real data remain ordinary zero-filled inputs.
    This generator neither adds acquisition masks nor creates imputation targets.
    """

    def __init__(self, catalogue: IonCatalogue | None, mass_axis: torch.Tensor,
                 schemas: Mapping[str, TargetSchema], config: SyntheticSpectrumConfig,
                 eligible_ions: tuple[int, ...] | None = None,
                 chemistry: Mapping[str, Any] | None = None,
                 peak_source: SyntheticPeakSource | None = None,
                 annotation_population: AnnotationPopulation | None = None,
                 mass_to_bin: Callable[[np.ndarray], np.ndarray] | None = None,
                 dtype: torch.dtype = torch.float32) -> None:
        if peak_source is None:
            if catalogue is None:
                raise ValueError("Synthetic pretraining requires catalogue or peak_source.")
            peak_source = CataloguePeakSource(catalogue)
        if catalogue is not None and catalogue.feature_count != peak_source.feature_count:
            raise ValueError("catalogue and peak_source dimensions disagree.")
        self.catalogue = catalogue
        self.peak_source, self.config, self.dtype = peak_source, config, dtype
        self.schemas, self.chemistry = dict(schemas), dict(chemistry or {})
        self.mass_to_bin = mass_to_bin
        self.space = SpectrumSpace(mass_axis.detach().cpu(), normalization=config.normalization)
        if self.space.feature_count != peak_source.feature_count:
            raise ValueError("Synthetic axis and peak source dimensions disagree.")
        target_indices = {
            name: index
            for index, name in enumerate(self.schemas["molecule"].class_names)
        }
        self.source_to_target = tuple(
            target_indices.get(name) for name in peak_source.class_names
        )
        if config.label_targets and any(index is None for index in self.source_to_target):
            raise ValueError(
                "Labelled synthetic sources must be a subset of the molecule target vocabulary."
            )
        self.eligible_ions = tuple(range(len(peak_source.bins))) if eligible_ions is None else eligible_ions
        if any(i < 0 or i >= len(peak_source.bins) for i in self.eligible_ions):
            raise ValueError("Invalid synthetic ion index.")
        # Unannotated geometry cannot accidentally land inside a labelled peak window
        occupied = np.zeros(peak_source.feature_count, dtype=bool)
        for bins in peak_source.bins:
            for coordinate in bins:
                occupied[max(0, coordinate - 2 * config.peak_radius):coordinate + 2 * config.peak_radius + 1] = True
        self.background_bins = np.flatnonzero(~occupied)
        self.sampling_context = SyntheticSamplingContext(
            source=peak_source,
            eligible_labels=self.eligible_ions,
            background_bins=self.background_bins,
            default_max_peaks=config.max_peaks,
            annotation_population=annotation_population,
        )
        if annotation_population is not None and annotation_population.feature_count != peak_source.feature_count:
            raise ValueError("Annotation population and synthetic source dimensions disagree.")
        self._plan_entries = self._build_plan_entries(config)
        self._plan_entry_offsets: dict[int, int] = {}
        for plan_index, entry in enumerate(self._plan_entries):
            self._plan_entry_offsets.setdefault(id(entry), plan_index)
        self._strategies = {
            id(entry): get_sampling_strategy(entry.strategy, **entry.parameters)
            for entry in self._plan_entries
        }
        representation_specs = [config.representation]
        representation_specs.extend(
            SyntheticRepresentationSpec.from_value(entry.representation)
            for entry in self._plan_entries
            if entry.representation is not None
        )
        self._representations = {
            _representation_key(spec): get_representation_strategy(
                spec.strategy,
                **spec.parameters,
            )
            for spec in representation_specs
        }
        self._requested_labels_by_sample: dict[int, tuple[int, ...]] = {}
        self.epoch = 0
        self._prepare_annotation_schedule()
        logger.info("Synthetic pretraining dataset: samples=%s labels=%s bins=%s strategies=%s.",
                    config.samples, len(self.eligible_ions), peak_source.feature_count,
                    tuple(entry.strategy for entry in self._plan_entries))

    def __len__(self) -> int:
        return self.config.samples

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic epoch stream; validation datasets retain epoch zero."""
        if epoch < 0:
            raise ValueError("epoch must be nonnegative.")
        self.epoch = int(epoch)
        self._prepare_annotation_schedule()

    def _prepare_annotation_schedule(self) -> None:
        """Allocate deterministic, near-exact marginal class requests per epoch.

        Complete source-local records can carry secondary overlapping labels, so
        the final BCE-positive marginal is audited separately. This scheduler
        makes the requested primary labels differ by at most one occurrence.
        """
        self._requested_labels_by_sample = {}
        population = self.sampling_context.annotation_population
        if population is None or not self.config.sampling_plan:
            return
        offset = 0
        for entry in self.config.sampling_plan:
            output_count = entry.output_count
            if entry.strategy not in {
                "annotation_uniform_mixture",
                "annotation_rare_class_mixture",
            }:
                offset += output_count
                continue
            labels = (
                population.rare_labels(float(entry.parameters.get("rare_fraction", 0.25)))
                if entry.strategy == "annotation_rare_class_mixture"
                else population.positive_labels
            )
            minimum = int(entry.parameters.get("min_peaks", 15))
            maximum = int(entry.parameters.get("max_peaks", 30))
            if minimum < 1 or maximum < minimum:
                raise ValueError("Annotation mixture peak bounds are invalid.")
            counts = []
            for sample_index in range(offset, offset + output_count):
                generator = np.random.default_rng(
                    np.random.SeedSequence([self.config.seed, self.epoch, sample_index])
                )
                counts.append(int(generator.integers(minimum, maximum + 1)))
            total_requests = sum(counts)
            permutation_rng = np.random.default_rng(
                np.random.SeedSequence([self.config.seed, self.epoch, offset, 991])
            )
            cycle = tuple(int(value) for value in permutation_rng.permutation(labels))
            requests = tuple(cycle[index % len(cycle)] for index in range(total_requests))
            cursor = 0
            for sample_index, count in zip(range(offset, offset + output_count), counts, strict=True):
                self._requested_labels_by_sample[sample_index] = requests[cursor:cursor + count]
                cursor += count
            offset += output_count

    @staticmethod
    def _build_plan_entries(config: SyntheticSpectrumConfig) -> tuple[SyntheticSamplingPlanEntry, ...]:
        """Expand exact plans and candidate permutation variants."""
        if config.sampling_plan:
            return tuple(
                entry
                for entry in config.sampling_plan
                for _ in range(entry.output_count)
            )
        return tuple(
            SyntheticSamplingPlanEntry(strategy=mode, count=1)
            for mode in config.modes
        )

    def _sample_entry(
        self,
        index: int,
        rng: np.random.Generator,
    ) -> tuple[SyntheticSamplingPlanEntry, int, int]:
        """Select one planned strategy while retaining legacy weighted modes."""
        if self.config.sampling_plan:
            entry = self._plan_entries[index]
            first_index = self._plan_entry_offsets[id(entry)]
            return entry, index - first_index, entry.output_count
        entry = self._plan_entries[int(rng.integers(len(self._plan_entries)))]
        return entry, index, self.config.samples

    def __getitem__(self, index: int):
        """Return a synthetic spectrum and its complete ion-presence target."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, self.epoch, index]))
        entry, entry_index, entry_count = self._sample_entry(index, rng)
        context = replace(
            self.sampling_context,
            sample_index=index,
            entry_index=entry_index,
            entry_count=entry_count,
            epoch=self.epoch,
            requested_labels=self._requested_labels_by_sample.get(index, ()),
        )
        definition = self._strategies[id(entry)].sample(
            rng,
            context,
            label_targets=self.config.label_targets if entry.label_targets is None else entry.label_targets,
        )
        ions = [
            label
            for component in definition.components
            for label in component.label_indices
        ]
        representation_spec = (
            SyntheticRepresentationSpec.from_value(entry.representation)
            if entry.representation is not None
            else self.config.representation
        )
        renderer = self._representations[_representation_key(representation_spec)]
        spectrum = renderer.render(
            rng,
            SyntheticRepresentationContext(
                source=self.peak_source,
                feature_count=self.peak_source.feature_count,
                mass_axis=self.space.mass_axis.numpy(),
                mass_to_bin=self.mass_to_bin,
            ),
            definition,
        )  # (M,)
        if spectrum.shape != (self.peak_source.feature_count,) or bool((spectrum < 0).any()):
            raise ValueError("Synthetic representation must return a nonnegative vector on the active axis.")
        # Normalize the representation after the complete synthetic mixture is rendered.
        denominator = {"tic": spectrum.sum, "max": spectrum.max,
                       "l2": lambda: np.linalg.norm(spectrum), "none": lambda: 1.0}[self.config.normalization]()
        spectrum /= max(float(denominator), np.finfo(np.float64).tiny)

        values = {name: torch.zeros(schema.class_count, dtype=torch.float32) for name, schema in self.schemas.items()}
        masks = {name: torch.zeros(schema.class_count, dtype=torch.bool) for name, schema in self.schemas.items()}
        masks[simulated_negative_mask_key("molecule")] = torch.zeros(
            self.schemas["molecule"].class_count,
            dtype=torch.bool,
        )  # (C_molecule,)
        if definition.label_targets:
            target_ions = [self.source_to_target[ion] for ion in ions]
            if any(ion is None for ion in target_ions):
                raise ValueError(
                    "A labelled synthetic component has no molecule target index."
                )
            values["molecule"][[int(ion) for ion in target_ions]] = 1.0
            masks["molecule"].fill_(True)
            # Synthetic composition gives exact absences for its known catalogue.
            ## The target values remain zero; only this mask makes them N_sim.
            masks[simulated_negative_mask_key("molecule")].fill_(True)
            masks[simulated_negative_mask_key("molecule")][
                values["molecule"] > 0.5
            ] = False
        if definition.label_targets and "chemical_class" in values:
            class_index = {name: i for i, name in enumerate(self.schemas["chemical_class"].class_names)}
            certain, possible, complete = set(), set(), True
            for ion in ions:
                record = self.chemistry.get(self.peak_source.class_names[ion], {})
                certain.update(record.get("certain_classes", ()))
                possible.update(record.get("possible_classes", ()))
                complete &= bool(record.get("candidates")) and all(c.get("classes") for c in record.get("candidates", ()))
            for name in certain:
                if name in class_index:
                    values["chemical_class"][class_index[name]] = 1
            if complete:
                masks["chemical_class"].fill_(True)
                for name in possible - certain:
                    if name in class_index:
                        masks["chemical_class"][class_index[name]] = False
            masks["chemical_class"] |= values["chemical_class"].bool()
        if definition.label_targets and "element_counts" in values and len(ions) == 1 and len(definition.components) == 1:
            counts = parse_formula(self.peak_source.class_names[ions[0]].split("|", 1)[0])
            values["element_counts"] = torch.tensor([counts.get(e, 0) for e in self.schemas["element_counts"].class_names], dtype=torch.float32)
            masks["element_counts"].fill_(True)
        return SyntheticSpectrumSample(
            sample_id=index,
            spectrum=torch.as_tensor(spectrum, dtype=self.dtype),
            values=values,
            masks=masks,
            metadata=dict(definition.metadata),
        )

    def collate_fn(self, samples) -> SpectrumBatch:
        """Collate dense synthetic examples using the existing model batch contract."""
        generation_metadata = tuple(dict(sample.metadata) for sample in samples)
        return SpectrumBatch(
            sample_ids=torch.tensor([s[0] for s in samples]),
            spectra=torch.stack([s[1] for s in samples]),  # (B, M)
            space=self.space,
            targets=TargetBatch(
                values={name: torch.stack([s[2][name] for s in samples]) for name in self.schemas},
                masks={
                    name: torch.stack([s[3][name] for s in samples])
                    for name in samples[0][3]
                },
                schemas=self.schemas,
            ),
            metadata=(
                {"synthetic_generation": generation_metadata}
                if any(generation_metadata)
                else {}
            ),
        )


def build_synthetic_partitions(dataset: Any, parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Create synthetic train/validation data from a train-only ion selection.

    :param dataset: Real dataset owning the reference split and target vocabulary.
    :param parameters: Generator settings plus ``validation_samples``.
    :return: Synthetic train and validation datasets, with no synthetic test split.
    """
    from ...training.criterions.autoencoder.head.training_targets import collect_training_multilabel_targets
    options = dict(parameters)
    validation_samples = options.pop("validation_samples", 512)
    peak_source_mode = str(options.pop("peak_source", "annotation"))
    candidate_filters = options.pop("candidate_filters", None)
    candidate_classes = options.pop("candidate_classes", None)
    candidate_label_targets = bool(options.pop("candidate_label_targets", True))
    if peak_source_mode not in {"annotation", "candidate_catalog"}:
        raise ValueError("peak_source must be 'annotation' or 'candidate_catalog'.")
    if peak_source_mode == "candidate_catalog" and not candidate_label_targets:
        options["label_targets"] = False
    config = SyntheticSpectrumConfig(**options)
    catalogue = IonCatalogue.from_dataset(dataset)
    context_getter = getattr(dataset, "get_synthetic_context", None)
    context = context_getter() if callable(context_getter) else dataset.active_context
    axis = torch.as_tensor(context.binner.GetXAxis(), dtype=torch.float64)
    schemas = dataset.get_target_schemas()
    chemistry = dataset.get_chemical_descriptions() if "chemical_class" in schemas else {}
    if peak_source_mode == "candidate_catalog":
        candidate_catalog = context.candidate_catalog
        if candidate_catalog is None:
            raise ValueError("Candidate synthesis requires an active candidate catalogue.")
        peak_source = CandidateCatalogPeakSource(
            candidate_catalog,
            context.binner,
            filters=candidate_filters,
            allowed_labels=(
                schemas["molecule"].class_names if candidate_label_targets else None
            ),
            chemical_classes=candidate_classes,
        )
        eligible = tuple(range(len(peak_source.bins)))
    else:
        targets, mask = collect_training_multilabel_targets(dataset, "molecule")
        eligible = tuple(
            ((targets > 0.5) & mask).any(dim=0).nonzero(as_tuple=True)[0].tolist()
        )
        peak_source = None
    annotation_population = (
        AnnotationPopulation.from_dataset(dataset)
        if peak_source_mode == "annotation"
        else None
    )
    mapper = getattr(context.binner, "map_mass_values_to_bins", None)
    # REMARK: Legacy peak-profile renderers operate only on bin indices.  An
    # isotope renderer receives the mapper when the active binner exposes it;
    # it validates the requirement when selected rather than making legacy
    # synthetic phases depend on a newer binner interface.
    if not callable(mapper):
        mapper = None
    common = dict(catalogue=catalogue, mass_axis=axis, schemas=schemas,
                  eligible_ions=eligible, chemistry=chemistry, peak_source=peak_source,
                  annotation_population=annotation_population,
                  mass_to_bin=mapper,
                  dtype=getattr(dataset, "dtype", torch.float32))
    validation_options = {**options, "samples": validation_samples, "seed": config.seed + 1}
    if config.sampling_plan:
        validation_options["sampling_plan"] = _scale_sampling_plan(
            config.sampling_plan,
            validation_samples,
        )
    validation_config = SyntheticSpectrumConfig(**validation_options)
    return {"train": SyntheticSpectrumDataset(config=config, **common),
            "validation": SyntheticSpectrumDataset(config=validation_config, **common), "test": None}


def _scale_sampling_plan(
    plan: tuple[SyntheticSamplingPlanEntry, ...],
    sample_count: int,
) -> tuple[SyntheticSamplingPlanEntry, ...]:
    """Scale exact training proportions to the synthetic validation population.

    Largest-remainder allocation preserves the requested strategy proportions
    while making the resulting integer counts sum exactly to ``sample_count``.
    Entries assigned zero examples are omitted from the smaller validation plan.
    """
    source_count = sum(entry.output_count for entry in plan)
    expected = [entry.output_count * sample_count / source_count for entry in plan]
    allocated = [int(value) for value in expected]
    remainder = sample_count - sum(allocated)
    for index in sorted(
        range(len(plan)),
        key=lambda index: (expected[index] - allocated[index], -index),
        reverse=True,
    )[:remainder]:
        allocated[index] += 1
    scaled = []
    for entry, count in zip(plan, allocated):
        if count < 1:
            continue
        permutations = entry.parameters.get("permutations", 1)
        if entry.strategy == "candidate_permuted" and count % permutations == 0:
            scaled.append(replace(entry, count=count // permutations))
            continue
        parameters = dict(entry.parameters)
        parameters.pop("permutations", None)
        scaled.append(replace(entry, count=count, parameters=parameters))
    return tuple(scaled)


def _representation_key(specification: SyntheticRepresentationSpec) -> str:
    """Return a deterministic cache key for one YAML-compatible renderer spec."""
    return json.dumps(
        {
            "strategy": specification.strategy,
            "parameters": specification.parameters,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
