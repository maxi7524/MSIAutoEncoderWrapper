"""Reproducible synthetic spectra with complete generator-owned ion labels."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from msi_dataset_manager.annotations.chemistry import parse_formula

from ..annotation_evidence import IonCatalogue
from ..batches import SpectrumBatch
from ..spaces import SpectrumSpace
from ..targets import TargetBatch, TargetSchema
from ...utils.logger import get_custom_logger
from .sampling import (
    SyntheticSamplingContext,
    SyntheticSamplingPlanEntry,
    get_sampling_strategy,
)
from .sources import CataloguePeakSource, SyntheticPeakSource

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
        if normalized_plan:
            if sum(entry.count for entry in normalized_plan) != self.samples:
                raise ValueError("sampling_plan counts must sum exactly to samples.")
            for entry in normalized_plan:
                get_sampling_strategy(entry.strategy, **entry.parameters)
        elif not self.modes:
            raise ValueError("At least one legacy mode or sampling_plan entry is required.")
        else:
            for mode in self.modes:
                get_sampling_strategy(mode)
        if self.normalization not in {"tic", "max", "l2", "none"}:
            raise ValueError("Unsupported synthetic normalization.")


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
        self.space = SpectrumSpace(mass_axis.detach().cpu(), normalization=config.normalization)
        if self.space.feature_count != peak_source.feature_count:
            raise ValueError("Synthetic axis and peak source dimensions disagree.")
        if self.schemas["molecule"].class_names != peak_source.class_names:
            raise ValueError("Synthetic targets must preserve the real ion vocabulary order.")
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
        )
        self._plan_entries = self._build_plan_entries(config)
        self._strategies = {
            id(entry): get_sampling_strategy(entry.strategy, **entry.parameters)
            for entry in self._plan_entries
        }
        self.epoch = 0
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

    @staticmethod
    def _build_plan_entries(config: SyntheticSpectrumConfig) -> tuple[SyntheticSamplingPlanEntry, ...]:
        """Expand exact plans or preserve legacy weighted random mode selection."""
        if config.sampling_plan:
            return tuple(
                entry
                for entry in config.sampling_plan
                for _ in range(entry.count)
            )
        return tuple(
            SyntheticSamplingPlanEntry(strategy=mode, count=1)
            for mode in config.modes
        )

    def _sample_entry(self, index: int, rng: np.random.Generator) -> SyntheticSamplingPlanEntry:
        """Select one planned strategy while retaining legacy weighted modes."""
        if self.config.sampling_plan:
            return self._plan_entries[index]
        return self._plan_entries[int(rng.integers(len(self._plan_entries)))]

    def __getitem__(self, index: int):
        """Return a synthetic spectrum and its complete ion-presence target."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, self.epoch, index]))
        entry = self._sample_entry(index, rng)
        definition = self._strategies[id(entry)].sample(
            rng,
            self.sampling_context,
            label_targets=self.config.label_targets if entry.label_targets is None else entry.label_targets,
        )
        ions = [component.label_index for component in definition.components if component.label_index is not None]
        spectrum = np.zeros(self.peak_source.feature_count, dtype=np.float64)  # (M,)
        # Render controlled peak profiles and normalize the resulting mixture
        radius = self.config.peak_radius
        for component in definition.components:
            center = component.center
            left, right = max(0, center - radius), min(len(spectrum), center + radius + 1)
            position = np.arange(left, right)  # (W,)
            profile = 1 - np.abs(position - center) / (radius + 1)  # (W,)
            spectrum[left:right] += rng.uniform(0.1, 1.0) * profile
        denominator = {"tic": spectrum.sum, "max": spectrum.max,
                       "l2": lambda: np.linalg.norm(spectrum), "none": lambda: 1.0}[self.config.normalization]()
        spectrum /= max(float(denominator), np.finfo(np.float64).tiny)

        values = {name: torch.zeros(schema.class_count, dtype=torch.float32) for name, schema in self.schemas.items()}
        masks = {name: torch.zeros(schema.class_count, dtype=torch.bool) for name, schema in self.schemas.items()}
        if definition.label_targets:
            values["molecule"][ions] = 1.0
            masks["molecule"].fill_(True)
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
        return index, torch.as_tensor(spectrum, dtype=self.dtype), values, masks

    def collate_fn(self, samples) -> SpectrumBatch:
        """Collate dense synthetic examples using the existing model batch contract."""
        return SpectrumBatch(
            sample_ids=torch.tensor([s[0] for s in samples]),
            spectra=torch.stack([s[1] for s in samples]),  # (B, M)
            space=self.space,
            targets=TargetBatch(
                values={name: torch.stack([s[2][name] for s in samples]) for name in self.schemas},
                masks={name: torch.stack([s[3][name] for s in samples]) for name in self.schemas},
                schemas=self.schemas,
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
    config = SyntheticSpectrumConfig(**options)
    catalogue = IonCatalogue.from_dataset(dataset)
    targets, mask = collect_training_multilabel_targets(dataset, "molecule")
    eligible = tuple(((targets > 0.5) & mask).any(dim=0).nonzero(as_tuple=True)[0].tolist())
    axis = torch.as_tensor(dataset.active_context.binner.GetXAxis(), dtype=torch.float64)
    schemas = dataset.get_target_schemas()
    chemistry = dataset.get_chemical_descriptions() if "chemical_class" in schemas else {}
    common = dict(catalogue=catalogue, mass_axis=axis, schemas=schemas,
                  eligible_ions=eligible, chemistry=chemistry, dtype=getattr(dataset, "dtype", torch.float32))
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
    source_count = sum(entry.count for entry in plan)
    expected = [entry.count * sample_count / source_count for entry in plan]
    allocated = [int(value) for value in expected]
    remainder = sample_count - sum(allocated)
    for index in sorted(
        range(len(plan)),
        key=lambda index: (expected[index] - allocated[index], -index),
        reverse=True,
    )[:remainder]:
        allocated[index] += 1
    return tuple(
        replace(entry, count=count)
        for entry, count in zip(plan, allocated)
        if count > 0
    )
