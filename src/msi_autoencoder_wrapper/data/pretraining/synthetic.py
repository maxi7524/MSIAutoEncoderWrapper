"""Reproducible synthetic spectra with complete generator-owned ion labels."""

from __future__ import annotations

from dataclasses import dataclass
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

logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class SyntheticSpectrumConfig:
    """Control the geometry and sampling of a synthetic pretraining population.

    :param samples: Number of generated spectra per epoch.
    :param seed: Local seed, independent of global NumPy and Torch state.
    :param max_peaks: Upper bound on sampled components per spectrum.
    :param modes: Mixture modes; repeated names increase their sampling frequency.
    :param peak_radius: Half-width of a triangular peak in bins; zero is a point.
    :param normalization: ``tic``, ``max``, ``l2``, or ``none``.
    """

    samples: int = 4096
    seed: int = 42
    max_peaks: int = 16
    modes: tuple[str, ...] = ("single_random", "single_annotated", "random", "annotated", "mixed")
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
        if not self.modes or set(self.modes) - {"single_random", "single_annotated", "random", "annotated", "mixed"}:
            raise ValueError("Unsupported synthetic sampling mode.")
        if self.normalization not in {"tic", "max", "l2", "none"}:
            raise ValueError("Unsupported synthetic normalization.")


class SyntheticSpectrumDataset(Dataset):
    """Generate examples by index and epoch, without reading production spectra.

    :param catalogue: Ion labels and their binned coordinates.
    :param mass_axis: Global model axis, shape ``(M,)``.
    :param schemas: Shared real/synthetic target schemas.
    :param config: Generator configuration.
    :param eligible_ions: Optional training-only ion indices.
    :param chemistry: Optional frozen ion descriptions.
    :param dtype: Numerical dtype of generated spectra.

    REMARK: Missing ranges in real data remain ordinary zero-filled inputs.
    This generator neither adds acquisition masks nor creates imputation targets.
    """

    def __init__(self, catalogue: IonCatalogue, mass_axis: torch.Tensor,
                 schemas: Mapping[str, TargetSchema], config: SyntheticSpectrumConfig,
                 eligible_ions: tuple[int, ...] | None = None,
                 chemistry: Mapping[str, Any] | None = None,
                 dtype: torch.dtype = torch.float32) -> None:
        self.catalogue, self.config, self.dtype = catalogue, config, dtype
        self.schemas, self.chemistry = dict(schemas), dict(chemistry or {})
        self.space = SpectrumSpace(mass_axis.detach().cpu(), normalization=config.normalization)
        if self.space.feature_count != catalogue.feature_count:
            raise ValueError("Synthetic axis and catalogue dimensions disagree.")
        if self.schemas["molecule"].class_names != catalogue.class_names:
            raise ValueError("Synthetic targets must preserve the real ion vocabulary order.")
        self.eligible_ions = tuple(range(len(catalogue.bins))) if eligible_ions is None else eligible_ions
        if any(i < 0 or i >= len(catalogue.bins) for i in self.eligible_ions):
            raise ValueError("Invalid synthetic ion index.")
        if not self.eligible_ions and any(mode in {"single_annotated", "annotated", "mixed"} for mode in config.modes):
            raise ValueError("Annotated synthesis requires positive ions in the training partition.")
        # Unannotated geometry cannot accidentally land inside a labelled peak window
        occupied = np.zeros(catalogue.feature_count, dtype=bool)
        for bins in catalogue.bins:
            for coordinate in bins:
                occupied[max(0, coordinate - 2 * config.peak_radius):coordinate + 2 * config.peak_radius + 1] = True
        self.background_bins = np.flatnonzero(~occupied)
        if not len(self.background_bins) and any(mode in {"single_random", "random", "mixed"} for mode in config.modes):
            raise ValueError("No unannotated bins remain for the requested synthetic modes.")
        self.epoch = 0
        logger.info("Synthetic pretraining dataset: samples=%s ions=%s bins=%s modes=%s.",
                    config.samples, len(self.eligible_ions), catalogue.feature_count, config.modes)

    def __len__(self) -> int:
        return self.config.samples

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic epoch stream; validation datasets retain epoch zero."""
        if epoch < 0:
            raise ValueError("epoch must be nonnegative.")
        self.epoch = int(epoch)

    def __getitem__(self, index: int):
        """Return a synthetic spectrum and its complete ion-presence target."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, self.epoch, index]))
        mode = self.config.modes[int(rng.integers(len(self.config.modes)))]
        count = 1 if mode.startswith("single_") else int(rng.integers(1, self.config.max_peaks + 1))
        ion_count = (0 if mode in {"single_random", "random"} else count
                     if mode in {"single_annotated", "annotated"} else int(rng.integers(count + 1)))
        ion_count = min(ion_count, len(self.eligible_ions))
        ions = rng.choice(self.eligible_ions, size=ion_count, replace=False).tolist()
        background_count = min(count - ion_count, len(self.background_bins)) if mode not in {"annotated", "single_annotated"} else 0
        centers = [int(rng.choice(self.catalogue.bins[i])) for i in ions]
        centers += rng.choice(self.background_bins, size=background_count, replace=False).tolist()
        spectrum = np.zeros(self.catalogue.feature_count, dtype=np.float64)  # (M,)
        # Render controlled peak profiles and normalize the resulting mixture
        radius = self.config.peak_radius
        for center in centers:
            left, right = max(0, center - radius), min(len(spectrum), center + radius + 1)
            position = np.arange(left, right)  # (W,)
            profile = 1 - np.abs(position - center) / (radius + 1)  # (W,)
            spectrum[left:right] += rng.uniform(0.1, 1.0) * profile
        denominator = {"tic": spectrum.sum, "max": spectrum.max,
                       "l2": lambda: np.linalg.norm(spectrum), "none": lambda: 1.0}[self.config.normalization]()
        spectrum /= max(float(denominator), np.finfo(np.float64).tiny)

        values = {name: torch.zeros(schema.class_count, dtype=torch.float32) for name, schema in self.schemas.items()}
        masks = {name: torch.zeros(schema.class_count, dtype=torch.bool) for name, schema in self.schemas.items()}
        values["molecule"][ions] = 1.0
        masks["molecule"].fill_(True)
        if "chemical_class" in values:
            class_index = {name: i for i, name in enumerate(self.schemas["chemical_class"].class_names)}
            certain, possible, complete = set(), set(), True
            for ion in ions:
                record = self.chemistry.get(self.catalogue.class_names[ion], {})
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
        if "element_counts" in values and len(ions) == 1 and background_count == 0:
            counts = parse_formula(self.catalogue.class_names[ions[0]].split("|", 1)[0])
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
    validation_config = SyntheticSpectrumConfig(**{**options, "samples": validation_samples, "seed": config.seed + 1})
    return {"train": SyntheticSpectrumDataset(config=config, **common),
            "validation": SyntheticSpectrumDataset(config=validation_config, **common), "test": None}
