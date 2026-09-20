"""Configuration and compilation of persistent synthetic pretraining artifacts.

An artifact stores immutable source geometry, sparse component profiles, and
compact sample manifests. Dense spectra and target tensors are created only for
the requested DataLoader batch.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from ..batches import SpectrumBatch
from ..spaces import SpectrumSpace
from ..supervision_masks import simulated_negative_mask_key
from ..targets import TargetBatch, TargetSchema
from ...utils.logger import get_custom_logger
from .representations import (
    SyntheticRepresentationContext,
    SyntheticRepresentationSpec,
    get_representation_strategy,
)
from .sampling import SyntheticComponent, SyntheticSampleDefinition
from .sources import CandidateCatalogPeakSource, SyntheticPeakSource

logger = get_custom_logger(__name__)

_ARTIFACT_SCHEMA_VERSION = 1
_EMPTY_COMPONENT = -1


@dataclass(frozen=True)
class PrecomputedPopulationSpec:
    """One synthetic population compiled into a static artifact manifest."""

    name: str
    strategy: str
    repetitions_per_bin: int
    min_fragments: int | None = None
    max_fragments: int | None = None
    detection_probability: float = 1.0

    @classmethod
    def from_mapping(
        cls,
        name: str,
        value: Mapping[str, Any],
    ) -> "PrecomputedPopulationSpec":
        """Validate one YAML-compatible population declaration."""
        if not isinstance(value, Mapping):
            raise ValueError("Every precomputed population must be a mapping.")
        strategy = value.get("strategy")
        repetitions = value.get("repetitions_per_bin")
        minimum = value.get("min_fragments")
        maximum = value.get("max_fragments")
        probability = value.get("detection_probability", 1.0)
        if strategy not in {"axis_coverage", "uniform_mixture"}:
            raise ValueError(
                "Population strategy must be 'axis_coverage' or 'uniform_mixture'."
            )
        if (
            isinstance(repetitions, bool)
            or not isinstance(repetitions, int)
            or repetitions < 1
        ):
            raise ValueError("repetitions_per_bin must be a positive integer.")
        if strategy == "axis_coverage":
            if minimum is not None or maximum is not None:
                raise ValueError(
                    "axis_coverage does not accept min_fragments or max_fragments."
                )
        elif (
            isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or minimum < 1
            or maximum < minimum
        ):
            raise ValueError(
                "uniform_mixture requires 1 <= min_fragments <= max_fragments."
            )
        if (
            not isinstance(probability, (int, float))
            or not 0.0 < float(probability) <= 1.0
        ):
            raise ValueError("detection_probability must belong to (0, 1].")
        unsupported = set(value).difference(
            {
                "strategy",
                "repetitions_per_bin",
                "min_fragments",
                "max_fragments",
                "detection_probability",
            }
        )
        if unsupported:
            raise ValueError(
                f"Unsupported precomputed population keys: {sorted(unsupported)}."
            )
        return cls(
            name=name,
            strategy=strategy,
            repetitions_per_bin=repetitions,
            min_fragments=minimum,
            max_fragments=maximum,
            detection_probability=float(probability),
        )


@dataclass(frozen=True)
class PrecomputedSyntheticConfig:
    """Immutable request for an artifact-backed synthetic population."""

    artifact_key: str
    cache_directory: Path
    peak_source: str
    representation: SyntheticRepresentationSpec
    normalization: str
    seed: int
    populations: tuple[PrecomputedPopulationSpec, ...]
    selected_population: str
    validation_samples: int
    blank_peak_radius: int
    candidate_filters: Mapping[str, Any] | None
    candidate_classes: tuple[str, ...] | None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "PrecomputedSyntheticConfig":
        """Parse an explicit precomputed_synthetic phase declaration."""
        if value.get("kind") != "precomputed_synthetic":
            raise ValueError(
                "Precomputed phases require kind='precomputed_synthetic'."
            )
        artifact = value.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ValueError("pretraining.artifact must be a mapping.")
        artifact_key = artifact.get("key")
        cache_directory = artifact.get("cache_directory")
        peak_source = artifact.get("peak_source", "annotation")
        populations_value = artifact.get("populations")
        if not isinstance(artifact_key, str) or not artifact_key:
            raise ValueError("pretraining.artifact.key must be a nonempty string.")
        if not isinstance(cache_directory, str) or not cache_directory:
            raise ValueError(
                "pretraining.artifact.cache_directory must be a nonempty path."
            )
        if peak_source not in {"annotation", "candidate_catalog"}:
            raise ValueError("pretraining.artifact.peak_source is unsupported.")
        if not isinstance(populations_value, Mapping) or not populations_value:
            raise ValueError(
                "pretraining.artifact.populations must be a nonempty mapping."
            )
        populations = tuple(
            PrecomputedPopulationSpec.from_mapping(str(name), population)
            for name, population in populations_value.items()
        )
        selected_population = value.get("population")
        if selected_population not in {population.name for population in populations}:
            raise ValueError(
                "pretraining.population must select an artifact population."
            )
        representation = SyntheticRepresentationSpec.from_value(
            artifact.get("representation")
        )
        get_representation_strategy(
            representation.strategy,
            **representation.parameters,
        )
        normalization = artifact.get("normalization", "tic")
        if normalization not in {"tic", "max", "l2", "none"}:
            raise ValueError("Unsupported precomputed synthetic normalization.")
        seed = artifact.get("seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(
                "pretraining.artifact.seed must be a nonnegative integer."
            )
        validation_samples = value.get("validation_samples", 512)
        if (
            isinstance(validation_samples, bool)
            or not isinstance(validation_samples, int)
            or validation_samples < 1
        ):
            raise ValueError(
                "pretraining.validation_samples must be a positive integer."
            )
        blank_peak_radius = artifact.get("blank_peak_radius", 0)
        if (
            isinstance(blank_peak_radius, bool)
            or not isinstance(blank_peak_radius, int)
            or blank_peak_radius < 0
        ):
            raise ValueError(
                "pretraining.artifact.blank_peak_radius must be a nonnegative integer."
            )
        candidate_filters = artifact.get("candidate_filters")
        if candidate_filters is not None and not isinstance(candidate_filters, Mapping):
            raise ValueError("candidate_filters must be a mapping when configured.")
        candidate_classes = artifact.get("candidate_classes")
        if candidate_classes is not None and (
            not isinstance(candidate_classes, Sequence)
            or isinstance(candidate_classes, str)
            or not all(isinstance(item, str) and item for item in candidate_classes)
        ):
            raise ValueError(
                "candidate_classes must be a sequence of nonempty strings."
            )
        unsupported = set(value).difference(
            {"kind", "artifact", "population", "validation_samples"}
        )
        if unsupported:
            raise ValueError(
                f"Unsupported precomputed synthetic keys: {sorted(unsupported)}."
            )
        return cls(
            artifact_key=artifact_key,
            cache_directory=Path(cache_directory),
            peak_source=str(peak_source),
            representation=representation,
            normalization=str(normalization),
            seed=seed,
            populations=populations,
            selected_population=str(selected_population),
            validation_samples=validation_samples,
            blank_peak_radius=blank_peak_radius,
            candidate_filters=(
                dict(candidate_filters) if candidate_filters is not None else None
            ),
            candidate_classes=(
                tuple(candidate_classes) if candidate_classes is not None else None
            ),
        )


@dataclass(frozen=True)
class SyntheticManifest:
    """Compact declarations for one complete precomputed population."""

    component_ids: np.ndarray
    blank_centers: np.ndarray
    anchor_bins: np.ndarray

    def __post_init__(self) -> None:
        if self.component_ids.ndim != 2:
            raise ValueError("Manifest component IDs must have shape (N, K).")
        sample_count = self.component_ids.shape[0]
        if self.blank_centers.shape != (sample_count,):
            raise ValueError("Manifest blank centers must have shape (N,).")
        if self.anchor_bins.shape != (sample_count,):
            raise ValueError("Manifest anchor bins must have shape (N,).")


@dataclass(frozen=True)
class SyntheticPrecomputeArtifact:
    """Persistent basis bank and compact population manifests."""

    fingerprint: str
    artifact_key: str
    axis: np.ndarray
    basis_rows: np.ndarray
    basis_columns: np.ndarray
    basis_values: np.ndarray
    prototype_target_indices: np.ndarray
    prototype_anchor_bins: np.ndarray
    manifests: Mapping[str, SyntheticManifest]
    blank_peak_radius: int
    normalization: str

    @property
    def feature_count(self) -> int:
        """Return active axis width."""
        return int(self.axis.size)

    @property
    def prototype_count(self) -> int:
        """Return immutable labelled component count."""
        return int(self.prototype_target_indices.size)


class _StaticPeakSource(SyntheticPeakSource):
    """In-memory train-only annotation source."""

    def __init__(
        self,
        *,
        feature_count: int,
        class_names: tuple[str, ...],
        bins: tuple[tuple[int, ...], ...],
    ) -> None:
        self._feature_count = feature_count
        self._class_names = class_names
        self._bins = bins

    @property
    def feature_count(self) -> int:
        """Return active axis width."""
        return self._feature_count

    @property
    def class_names(self) -> tuple[str, ...]:
        """Return source labels in target order."""
        return self._class_names

    @property
    def bins(self) -> tuple[tuple[int, ...], ...]:
        """Return source anchor bins."""
        return self._bins


class SyntheticPrecomputeBuilder:
    """Compile a source, sparse basis, and manifests once."""

    def __init__(
        self,
        dataset: Any,
        config: PrecomputedSyntheticConfig,
        *,
        fingerprint: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.context = self._get_context()
        self.schemas = dict(dataset.get_target_schemas())
        if "molecule" not in self.schemas:
            raise ValueError(
                "Synthetic precompute requires a molecular target schema."
            )
        self.axis = np.asarray(self.context.binner.GetXAxis(), dtype=np.float64)
        if (
            self.axis.ndim != 1
            or not self.axis.size
            or not np.isfinite(self.axis).all()
        ):
            raise ValueError(
                "Synthetic precompute requires a finite one-dimensional axis."
            )
        (
            self.source,
            self.source_to_target,
            self.source_signature,
        ) = self._build_source()
        (
            self.prototype_labels,
            self.prototype_centers,
            self.anchor_groups,
        ) = self._build_prototypes()
        self.fingerprint = fingerprint or self._fingerprint()

    def build(self) -> SyntheticPrecomputeArtifact:
        """Render basis prototypes once and compile all configured manifests."""
        basis_rows, basis_columns, basis_values = self._build_sparse_basis()
        manifests = {
            population.name: self._compile_manifest(population)
            for population in self.config.populations
        }
        return SyntheticPrecomputeArtifact(
            fingerprint=self.fingerprint,
            artifact_key=self.config.artifact_key,
            axis=self.axis,
            basis_rows=basis_rows,
            basis_columns=basis_columns,
            basis_values=basis_values,
            prototype_target_indices=self.prototype_labels,
            prototype_anchor_bins=self.prototype_centers,
            manifests=manifests,
            blank_peak_radius=self.config.blank_peak_radius,
            normalization=self.config.normalization,
        )

    def _get_context(self) -> Any:
        """Resolve binner and optional candidate-catalogue context."""
        getter = getattr(self.dataset, "get_synthetic_context", None)
        return getter() if callable(getter) else self.dataset.active_context

    def _build_source(
        self,
    ) -> tuple[SyntheticPeakSource, np.ndarray, Mapping[str, Any]]:
        """Resolve one source with a stable target-column projection."""
        target_names = self.schemas["molecule"].class_names
        target_lookup = {name: index for index, name in enumerate(target_names)}
        if self.config.peak_source == "annotation":
            index = self.dataset.get_mapped_annotation_index()
            train_ids = _partition_source_ids(self.dataset.create_partitions().train)
            bins = _compact_annotation_bins(
                index=index,
                selected_spectrum_ids=train_ids,
                target_indices=target_lookup,
                target_count=len(target_names),
            )
            source = _StaticPeakSource(
                feature_count=self.axis.size,
                class_names=tuple(target_names),
                bins=bins,
            )
            signature = {
                "kind": "annotation",
                "spectrum_ids": tuple(sorted(set(int(value) for value in train_ids))),
                "bins": bins,
            }
            return (
                source,
                np.arange(len(target_names), dtype=np.int32),
                signature,
            )
        candidate_catalog = getattr(self.context, "candidate_catalog", None)
        if candidate_catalog is None:
            raise ValueError(
                "Candidate synthetic precompute requires an active candidate catalogue."
            )
        source = CandidateCatalogPeakSource(
            candidate_catalog,
            self.context.binner,
            filters=self.config.candidate_filters,
            allowed_labels=target_names,
            chemical_classes=self.config.candidate_classes,
        )
        source_to_target = np.asarray(
            [target_lookup.get(name, -1) for name in source.class_names],
            dtype=np.int32,
        )
        if bool((source_to_target < 0).any()):
            raise ValueError(
                "Candidate source labels must be represented in molecular targets."
            )
        candidate_path = getattr(candidate_catalog, "path", None)
        path = Path(candidate_path) if candidate_path is not None else None
        signature = {
            "kind": "candidate_catalog",
            "path": str(path) if path is not None else None,
            "mtime_ns": path.stat().st_mtime_ns if path and path.exists() else None,
            "size": path.stat().st_size if path and path.exists() else None,
            "class_names": source.class_names,
            "bins": source.bins,
        }
        return source, source_to_target, signature

    def _build_prototypes(
        self,
    ) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
        """Assign one reusable prototype to every source label and anchor bin."""
        labels: list[int] = []
        centers: list[int] = []
        groups: list[list[int]] = [[] for _ in range(self.axis.size)]
        for source_label, coordinates in enumerate(self.source.bins):
            target_label = int(self.source_to_target[source_label])
            if target_label < 0:
                continue
            for coordinate in sorted(set(int(value) for value in coordinates)):
                if coordinate < 0 or coordinate >= self.axis.size:
                    raise ValueError(
                        "Synthetic source coordinate is outside the active axis."
                    )
                prototype_id = len(labels)
                labels.append(target_label)
                centers.append(coordinate)
                groups[coordinate].append(prototype_id)
        return (
            np.asarray(labels, dtype=np.int32),
            np.asarray(centers, dtype=np.int32),
            tuple(tuple(group) for group in groups),
        )

    def _fingerprint(self) -> str:
        """Return content identity for all immutable artifact inputs."""
        payload = {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "artifact_key": self.config.artifact_key,
            "peak_source": self.config.peak_source,
            "representation": {
                "strategy": self.config.representation.strategy,
                "parameters": self.config.representation.parameters,
            },
            "normalization": self.config.normalization,
            "blank_peak_radius": self.config.blank_peak_radius,
            "seed": self.config.seed,
            "populations": [
                {
                    "name": population.name,
                    "strategy": population.strategy,
                    "repetitions_per_bin": population.repetitions_per_bin,
                    "min_fragments": population.min_fragments,
                    "max_fragments": population.max_fragments,
                    "detection_probability": population.detection_probability,
                }
                for population in self.config.populations
            ],
            "target_names": self.schemas["molecule"].class_names,
            "source": self.source_signature,
        }
        digest = sha256()
        digest.update(
            json.dumps(payload, sort_keys=True, default=list).encode("utf-8")
        )
        digest.update(self.axis.tobytes())
        digest.update(self.prototype_labels.tobytes())
        digest.update(self.prototype_centers.tobytes())
        return digest.hexdigest()

    def _build_sparse_basis(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render every immutable prototype once and retain only nonzero bins."""
        renderer = get_representation_strategy(
            self.config.representation.strategy,
            **self.config.representation.parameters,
        )
        mapper = getattr(self.context.binner, "map_mass_values_to_bins", None)
        representation_context = SyntheticRepresentationContext(
            source=self.source,
            feature_count=self.axis.size,
            mass_axis=self.axis,
            mass_to_bin=mapper if callable(mapper) else None,
        )
        rows: list[np.ndarray] = []
        columns: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for prototype_id, (target_label, center) in enumerate(
            zip(
                self.prototype_labels,
                self.prototype_centers,
                strict=True,
            )
        ):
            source_label = self._source_label_for_target_center(
                int(target_label),
                int(center),
            )
            definition = SyntheticSampleDefinition(
                components=(
                    SyntheticComponent(
                        center=int(center),
                        label_index=source_label,
                        intensity_weight=1.0,
                    ),
                ),
                label_targets=True,
            )
            spectrum = np.asarray(
                renderer.render(
                    np.random.default_rng(0),
                    representation_context,
                    definition,
                ),
                dtype=np.float64,
            )
            if (
                spectrum.shape != (self.axis.size,)
                or not np.isfinite(spectrum).all()
                or bool((spectrum < 0).any())
            ):
                raise ValueError(
                    "Synthetic basis renderer returned an invalid spectrum."
                )
            total = float(spectrum.sum())
            if total <= 0:
                raise ValueError(
                    "Synthetic basis component has zero total intensity."
                )
            spectrum /= total
            nonzero = np.flatnonzero(spectrum > 0)
            rows.append(
                np.full(nonzero.size, prototype_id, dtype=np.int32)
            )
            columns.append(nonzero.astype(np.int32, copy=False))
            values.append(spectrum[nonzero].astype(np.float32, copy=False))
        if not rows:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )
        return (
            np.concatenate(rows),
            np.concatenate(columns),
            np.concatenate(values),
        )

    def _source_label_for_target_center(
        self,
        target_label: int,
        center: int,
    ) -> int:
        """Resolve source-local label for one target/anchor prototype."""
        matches = np.flatnonzero(self.source_to_target == target_label)
        for source_label in matches:
            if center in self.source.bins[int(source_label)]:
                return int(source_label)
        raise ValueError(
            "Prototype cannot be resolved to a source-local component."
        )

    def _compile_manifest(
        self,
        population: PrecomputedPopulationSpec,
    ) -> SyntheticManifest:
        """Compile one static axis or mixture population."""
        if population.strategy == "axis_coverage":
            return self._compile_axis_manifest(population)
        return self._compile_uniform_mixture_manifest(population)

    def _compile_axis_manifest(
        self,
        population: PrecomputedPopulationSpec,
    ) -> SyntheticManifest:
        """Compile exact full-axis repetitions with complete anchor groups."""
        anchors = np.tile(
            np.arange(self.axis.size, dtype=np.int32),
            population.repetitions_per_bin,
        )
        components = [self.anchor_groups[int(anchor)] for anchor in anchors]
        blank_centers = np.asarray(
            [
                anchor if not component_ids else -1
                for anchor, component_ids in zip(
                    anchors,
                    components,
                    strict=True,
                )
            ],
            dtype=np.int32,
        )
        return SyntheticManifest(
            component_ids=_pad_components(components),
            blank_centers=blank_centers,
            anchor_bins=anchors,
        )

    def _compile_uniform_mixture_manifest(
        self,
        population: PrecomputedPopulationSpec,
    ) -> SyntheticManifest:
        """Compile quota-satisfying mixtures of distinct nonempty anchor bins."""
        eligible_bins = np.asarray(
            [
                index
                for index, group in enumerate(self.anchor_groups)
                if group
            ],
            dtype=np.int32,
        )
        if eligible_bins.size < int(population.min_fragments):
            raise ValueError(
                "uniform_mixture requires at least min_fragments nonempty bins."
            )
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.config.seed, _stable_name_seed(population.name)]
            )
        )
        remaining = np.full(
            eligible_bins.size,
            population.repetitions_per_bin,
            dtype=np.int32,
        )
        all_positions = np.arange(eligible_bins.size, dtype=np.int32)
        component_rows: list[tuple[int, ...]] = []
        anchors: list[int] = []
        while bool(np.any(remaining > 0)):
            requested = int(
                rng.integers(
                    int(population.min_fragments),
                    int(population.max_fragments) + 1,
                )
            )
            active = all_positions[remaining > 0]
            random_order = rng.random(active.size)
            priority = np.lexsort((random_order, -remaining[active]))
            selected = active[priority[: min(requested, active.size)]]
            if selected.size < requested:
                available = np.setdiff1d(
                    all_positions,
                    selected,
                    assume_unique=True,
                )
                fill = rng.choice(
                    available,
                    size=requested - selected.size,
                    replace=False,
                )
                selected = np.concatenate((selected, fill))
            selected_bins = eligible_bins[selected]
            retained: list[int] = []
            for anchor in selected_bins:
                group = self.anchor_groups[int(anchor)]
                keep = rng.random(len(group)) < population.detection_probability
                if not bool(keep.any()):
                    keep[int(rng.integers(len(group)))] = True
                retained.extend(
                    prototype
                    for prototype, selected_component in zip(
                        group,
                        keep,
                        strict=True,
                    )
                    if selected_component
                )
            component_rows.append(tuple(retained))
            anchors.append(int(selected_bins[0]))
            selected_active = selected[remaining[selected] > 0]
            remaining[selected_active] -= 1
        return SyntheticManifest(
            component_ids=_pad_components(component_rows),
            blank_centers=np.full(len(component_rows), -1, dtype=np.int32),
            anchor_bins=np.asarray(anchors, dtype=np.int32),
        )


class SyntheticArtifactStore:
    """Persist immutable synthetic artifacts with atomic lock protection."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load_or_build(
        self,
        *,
        artifact_key: str,
        fingerprint: str,
        builder: Callable[[], SyntheticPrecomputeArtifact],
    ) -> SyntheticPrecomputeArtifact:
        """Load matching output or build it exactly once under a file lock."""
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{_safe_key(artifact_key)}-{fingerprint[:16]}"
        loaded = self._load(target, fingerprint)
        if loaded is not None:
            logger.info("Loaded synthetic precompute artifact: %s.", target)
            return loaded
        lock_path = self.root / (
            f".{_safe_key(artifact_key)}-{fingerprint[:16]}.lock"
        )
        with _artifact_lock(lock_path):
            loaded = self._load(target, fingerprint)
            if loaded is not None:
                logger.info(
                    "Loaded synthetic precompute artifact after lock: %s.",
                    target,
                )
                return loaded
            artifact = builder()
            temporary = self.root / f".{target.name}.{uuid4().hex}.tmp"
            temporary.mkdir(parents=True)
            try:
                self._save(temporary, artifact)
                os.replace(temporary, target)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            logger.info(
                "Built synthetic artifact: %s prototypes=%s populations=%s.",
                target,
                artifact.prototype_count,
                tuple(artifact.manifests),
            )
            return artifact

    def _load(
        self,
        directory: Path,
        fingerprint: str,
    ) -> SyntheticPrecomputeArtifact | None:
        """Load complete matching arrays, rejecting partial or stale artifacts."""
        metadata_path = directory / "metadata.json"
        arrays_path = directory / "artifact.npz"
        if not metadata_path.is_file() or not arrays_path.is_file():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("schema_version") != _ARTIFACT_SCHEMA_VERSION
            or metadata.get("fingerprint") != fingerprint
        ):
            return None
        with np.load(arrays_path, allow_pickle=False) as arrays:
            manifests = {
                name: SyntheticManifest(
                    component_ids=arrays[f"{name}__component_ids"],
                    blank_centers=arrays[f"{name}__blank_centers"],
                    anchor_bins=arrays[f"{name}__anchor_bins"],
                )
                for name in metadata["population_names"]
            }
            return SyntheticPrecomputeArtifact(
                fingerprint=str(metadata["fingerprint"]),
                artifact_key=str(metadata["artifact_key"]),
                axis=arrays["axis"],
                basis_rows=arrays["basis_rows"],
                basis_columns=arrays["basis_columns"],
                basis_values=arrays["basis_values"],
                prototype_target_indices=arrays["prototype_target_indices"],
                prototype_anchor_bins=arrays["prototype_anchor_bins"],
                manifests=manifests,
                blank_peak_radius=int(metadata["blank_peak_radius"]),
                normalization=str(metadata["normalization"]),
            )

    def _save(
        self,
        directory: Path,
        artifact: SyntheticPrecomputeArtifact,
    ) -> None:
        """Write artifact arrays before publishing their completion metadata."""
        arrays: dict[str, np.ndarray] = {
            "axis": artifact.axis,
            "basis_rows": artifact.basis_rows,
            "basis_columns": artifact.basis_columns,
            "basis_values": artifact.basis_values,
            "prototype_target_indices": artifact.prototype_target_indices,
            "prototype_anchor_bins": artifact.prototype_anchor_bins,
        }
        for name, manifest in artifact.manifests.items():
            arrays[f"{name}__component_ids"] = manifest.component_ids
            arrays[f"{name}__blank_centers"] = manifest.blank_centers
            arrays[f"{name}__anchor_bins"] = manifest.anchor_bins
        np.savez_compressed(directory / "artifact.npz", **arrays)
        metadata = {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "artifact_key": artifact.artifact_key,
            "fingerprint": artifact.fingerprint,
            "blank_peak_radius": artifact.blank_peak_radius,
            "normalization": artifact.normalization,
            "population_names": list(artifact.manifests),
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class PrecomputedSyntheticDataset(Dataset):
    """Expose selected manifest rows through the existing SpectrumBatch API."""

    def __init__(
        self,
        artifact: SyntheticPrecomputeArtifact,
        *,
        population: str,
        schemas: Mapping[str, TargetSchema],
        dtype: torch.dtype,
        row_indices: np.ndarray | None = None,
        fixed_epoch: bool = False,
        seed: int = 0,
    ) -> None:
        if population not in artifact.manifests:
            raise ValueError(f"Unknown synthetic artifact population: {population}.")
        self.artifact = artifact
        self.population = population
        self.manifest = artifact.manifests[population]
        self.schemas = dict(schemas)
        self.dtype = dtype
        self.fixed_epoch = fixed_epoch
        self.seed = int(seed)
        self.row_indices = (
            np.arange(self.manifest.component_ids.shape[0], dtype=np.int64)
            if row_indices is None
            else np.asarray(row_indices, dtype=np.int64)
        )
        if self.row_indices.ndim != 1 or not self.row_indices.size:
            raise ValueError("Synthetic dataset needs at least one manifest row.")
        if bool((self.row_indices < 0).any()) or bool(
            (self.row_indices >= self.manifest.component_ids.shape[0]).any()
        ):
            raise ValueError("Synthetic dataset indices are outside the manifest.")
        self.space = SpectrumSpace(
            torch.as_tensor(artifact.axis, dtype=torch.float64),
            normalization=artifact.normalization,
        )
        sparse_indices = (
            torch.as_tensor(
                np.stack((artifact.basis_rows, artifact.basis_columns)),
                dtype=torch.long,
            )
            if artifact.basis_rows.size
            else torch.empty((2, 0), dtype=torch.long)
        )
        self._basis = torch.sparse_coo_tensor(
            sparse_indices,
            torch.as_tensor(artifact.basis_values, dtype=torch.float32),
            size=(artifact.prototype_count, artifact.feature_count),
        ).coalesce()
        self._transposed_basis = self._basis.transpose(0, 1).coalesce()
        self._prototype_targets = torch.as_tensor(
            artifact.prototype_target_indices,
            dtype=torch.long,
        )
        self.epoch = 0

    def __len__(self) -> int:
        """Return selected manifest-row count."""
        return int(self.row_indices.size)

    def __getitem__(self, index: int) -> int:
        """Return lightweight manifest row ID for batch rendering."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return int(self.row_indices[index])

    def set_epoch(self, epoch: int) -> None:
        """Select per-epoch weights without rebuilding static population state."""
        if epoch < 0:
            raise ValueError("epoch must be nonnegative.")
        if not self.fixed_epoch:
            self.epoch = int(epoch)

    def collate_fn(self, rows: Sequence[int]) -> SpectrumBatch:
        """Vectorially render one batch and complete molecular targets."""
        row_ids = np.asarray(rows, dtype=np.int64)
        component_ids = self.manifest.component_ids[row_ids]  # (B, K)
        blank_centers = self.manifest.blank_centers[row_ids]  # (B,)
        spectra = self._render_batch(component_ids, blank_centers, row_ids)  # (B, M)
        targets = self._build_targets(component_ids)
        return SpectrumBatch(
            sample_ids=torch.as_tensor(row_ids, dtype=torch.long),
            spectra=spectra.to(dtype=self.dtype),
            space=self.space,
            targets=targets,
            metadata={
                "synthetic_precompute": {
                    "artifact_key": self.artifact.artifact_key,
                    "fingerprint": self.artifact.fingerprint,
                    "population": self.population,
                    "epoch": self.epoch,
                }
            },
        )

    def _render_batch(
        self,
        component_ids: np.ndarray,
        blank_centers: np.ndarray,
        row_ids: np.ndarray,
    ) -> torch.Tensor:
        """Render sparse basis mixtures and generic blank-bin peaks."""
        ids = torch.as_tensor(component_ids, dtype=torch.long)  # (B, K)
        valid = ids >= 0  # (B, K)
        batch_size = ids.shape[0]
        safe_ids = ids.clamp_min(0)  # (B, K)
        weights = _component_weights(
            row_ids=row_ids,
            component_mask=valid.numpy(),
            seed=self.seed,
            epoch=self.epoch,
        )  # (B, K)
        composition = torch.zeros(
            (batch_size, self.artifact.prototype_count),
            dtype=torch.float32,
        )  # (B, P)
        if self.artifact.prototype_count:
            composition.scatter_add_(1, safe_ids, weights)
            spectra = torch.sparse.mm(
                self._transposed_basis,
                composition.transpose(0, 1),
            ).transpose(0, 1)  # (B, M)
        else:
            spectra = torch.zeros(
                (batch_size, self.artifact.feature_count),
                dtype=torch.float32,
            )  # (B, M)
        _add_blank_profiles(
            spectra,
            torch.as_tensor(blank_centers, dtype=torch.long),
            radius=self.artifact.blank_peak_radius,
        )
        denominator = {
            "tic": spectra.sum(dim=1, keepdim=True),
            "max": spectra.amax(dim=1, keepdim=True),
            "l2": torch.linalg.vector_norm(spectra, dim=1, keepdim=True),
            "none": torch.ones((batch_size, 1), dtype=spectra.dtype),
        }[self.artifact.normalization]  # (B, 1)
        spectra = spectra / denominator.clamp_min(
            torch.finfo(spectra.dtype).tiny
        )  # (B, M)
        if not bool(torch.isfinite(spectra).all()) or bool((spectra < 0).any()):
            raise ValueError("Batch synthetic renderer produced invalid spectra.")
        return spectra

    def _build_targets(self, component_ids: np.ndarray) -> TargetBatch:
        """Create complete positives and known negatives for molecular targets."""
        batch_size = int(component_ids.shape[0])
        values: dict[str, torch.Tensor] = {
            name: torch.zeros(
                (batch_size, schema.class_count),
                dtype=torch.float32,
            )
            for name, schema in self.schemas.items()
        }
        masks: dict[str, torch.Tensor] = {
            name: torch.zeros(
                (batch_size, schema.class_count),
                dtype=torch.bool,
            )
            for name, schema in self.schemas.items()
        }
        ids = torch.as_tensor(component_ids, dtype=torch.long)  # (B, K)
        valid = ids >= 0  # (B, K)
        if self.artifact.prototype_count:
            labels = self._prototype_targets[ids.clamp_min(0)]  # (B, K)
            values["molecule"].scatter_add_(
                1,
                labels,
                valid.to(dtype=torch.float32),
            )
            values["molecule"].clamp_(0.0, 1.0)
        masks["molecule"].fill_(True)
        masks[simulated_negative_mask_key("molecule")] = (
            values["molecule"] < 0.5
        )  # (B, C_molecule)
        return TargetBatch(values=values, masks=masks, schemas=self.schemas)


# Public runtime implementations are isolated from compilation.  The local
# declarations above remain temporarily to preserve a reviewable move history;
# all builder and partition calls below resolve these dedicated implementations.
from .precompute_artifact import (
    ARTIFACT_SCHEMA_VERSION as _ARTIFACT_SCHEMA_VERSION,
    SyntheticArtifactStore,
    SyntheticManifest,
    SyntheticPrecomputeArtifact,
)
from .precompute_dataset import PrecomputedSyntheticDataset


def build_precomputed_synthetic_partitions(
    dataset: Any,
    parameters: Mapping[str, Any],
) -> dict[str, PrecomputedSyntheticDataset | None]:
    """Build or load artifact-backed train and validation datasets."""
    config = PrecomputedSyntheticConfig.from_mapping(parameters)
    fingerprint = precompute_request_fingerprint(dataset, config)

    def build_artifact() -> SyntheticPrecomputeArtifact:
        return SyntheticPrecomputeBuilder(
            dataset,
            config,
            fingerprint=fingerprint,
        ).build()

    artifact = SyntheticArtifactStore(config.cache_directory).load_or_build(
        artifact_key=config.artifact_key,
        fingerprint=fingerprint,
        builder=build_artifact,
    )
    manifest_size = artifact.manifests[
        config.selected_population
    ].component_ids.shape[0]
    validation_rows = _evenly_spaced_indices(
        manifest_size,
        min(config.validation_samples, manifest_size),
    )
    common = {
        "artifact": artifact,
        "population": config.selected_population,
        "schemas": dataset.get_target_schemas(),
        "dtype": getattr(dataset, "dtype", torch.float32),
        "seed": config.seed,
    }
    return {
        "train": PrecomputedSyntheticDataset(**common),
        "validation": PrecomputedSyntheticDataset(
            **common,
            row_indices=validation_rows,
            fixed_epoch=True,
        ),
        "test": None,
    }


def precompute_request_fingerprint(
    dataset: Any,
    config: PrecomputedSyntheticConfig,
) -> str:
    """Return a cheap cache identity without compiling source records or basis.

    The annotation branch hashes the immutable sparse index and selected train
    IDs directly. Therefore a cache hit avoids both train-record extraction and
    renderer construction; only the dataset's already-required sparse index is
    consulted.
    """
    context_getter = getattr(dataset, "get_synthetic_context", None)
    context = (
        context_getter() if callable(context_getter) else dataset.active_context
    )
    axis = np.asarray(context.binner.GetXAxis(), dtype=np.float64)
    schemas = dataset.get_target_schemas()
    digest = sha256()
    payload = {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "artifact_key": config.artifact_key,
        "peak_source": config.peak_source,
        "representation": {
            "strategy": config.representation.strategy,
            "parameters": config.representation.parameters,
        },
        "normalization": config.normalization,
        "blank_peak_radius": config.blank_peak_radius,
        "seed": config.seed,
        "populations": [
            {
                "name": population.name,
                "strategy": population.strategy,
                "repetitions_per_bin": population.repetitions_per_bin,
                "min_fragments": population.min_fragments,
                "max_fragments": population.max_fragments,
                "detection_probability": population.detection_probability,
            }
            for population in config.populations
        ],
        "target_names": schemas["molecule"].class_names,
    }
    if config.peak_source == "annotation":
        index = dataset.get_mapped_annotation_index()
        train_ids = _partition_source_ids(dataset.create_partitions().train)
        payload["annotation_identities"] = index.annotation_identities
        for values in (
            np.asarray(train_ids, dtype=np.int64),
            np.asarray(index.spectrum_ids, dtype=np.int64),
            np.asarray(index.spectrum_offsets, dtype=np.int64),
            np.asarray(index.annotation_indices, dtype=np.int64),
            np.asarray(index.coordinate_indices, dtype=np.int64),
        ):
            digest.update(values.tobytes())
    else:
        candidate_catalog = getattr(context, "candidate_catalog", None)
        if candidate_catalog is None:
            raise ValueError(
                "Candidate synthetic precompute requires an active candidate catalogue."
            )
        candidate_path = getattr(candidate_catalog, "path", None)
        path = Path(candidate_path) if candidate_path is not None else None
        payload["candidate_catalog"] = {
            "path": str(path) if path is not None else None,
            "mtime_ns": path.stat().st_mtime_ns if path and path.exists() else None,
            "size": path.stat().st_size if path and path.exists() else None,
            "filters": config.candidate_filters,
            "classes": config.candidate_classes,
        }
    digest.update(
        json.dumps(payload, sort_keys=True, default=list).encode("utf-8")
    )
    digest.update(axis.tobytes())
    return digest.hexdigest()


def _partition_source_ids(partition: Dataset) -> tuple[int, ...]:
    """Return stable source identifiers represented by a dataset partition."""
    if isinstance(partition, Subset):
        parent = _partition_source_ids(partition.dataset)
        return tuple(parent[int(index)] for index in partition.indices)
    source_ids_getter = getattr(partition, "get_sample_ids", None)
    if callable(source_ids_getter):
        return tuple(int(value) for value in source_ids_getter())
    return tuple(range(len(partition)))


def _compact_annotation_bins(
    *,
    index: Any,
    selected_spectrum_ids: Sequence[int],
    target_indices: Mapping[str, int],
    target_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Extract train-only target/bin geometry without materialising records.

    The sparse index can contain millions of annotation entries.  Precompute
    needs only the unique target/bin pairs, so retaining one Python record per
    entry would waste several gigabytes and can exceed the host memory limit.
    """
    requested = np.asarray(
        sorted({int(value) for value in selected_spectrum_ids}),
        dtype=np.int64,
    )
    indexed = np.asarray(index.spectrum_ids, dtype=np.int64)
    row_ids = np.searchsorted(indexed, requested)
    in_bounds = row_ids < indexed.size
    matched = row_ids[in_bounds]
    matched = matched[indexed[matched] == requested[in_bounds]]
    identity_targets = np.asarray(
        [
            target_indices.get("|".join(identity), -1)
            for identity in index.annotation_identities
        ],
        dtype=np.int32,
    )
    bins: list[set[int]] = [set() for _ in range(target_count)]
    for row_id in matched:
        start = int(index.spectrum_offsets[row_id])
        stop = int(index.spectrum_offsets[row_id + 1])
        entry_targets = identity_targets[index.annotation_indices[start:stop]]
        coordinates = index.coordinate_indices[start:stop]
        for target, coordinate in zip(entry_targets, coordinates, strict=True):
            if target >= 0:
                bins[int(target)].add(int(coordinate))
    return tuple(tuple(sorted(values)) for values in bins)


def _pad_components(rows: Sequence[Sequence[int]]) -> np.ndarray:
    """Pad variable-length rows with the empty-component sentinel."""
    width = max(1, max((len(row) for row in rows), default=0))
    result = np.full((len(rows), width), _EMPTY_COMPONENT, dtype=np.int32)
    for row_index, row in enumerate(rows):
        if len(set(row)) != len(row):
            raise ValueError(
                "Synthetic mixture contains duplicate component prototypes."
            )
        result[row_index, : len(row)] = row
    return result


def _stable_name_seed(value: str) -> int:
    """Return stable entropy for a population name."""
    return int.from_bytes(sha256(value.encode("utf-8")).digest()[:4], "big")


def _safe_key(value: str) -> str:
    """Normalize an artifact key into one filesystem-safe component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "synthetic"


@contextmanager
def _artifact_lock(path: Path, timeout_seconds: float = 300.0):
    """Serialize one artifact builder and prevent publication of partial data."""
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for synthetic artifact lock: {path}."
                )
            time.sleep(0.1)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _component_weights(
    *,
    row_ids: np.ndarray,
    component_mask: np.ndarray,
    seed: int,
    epoch: int,
) -> torch.Tensor:
    """Generate stateless vectorized Dirichlet(1) component weights."""
    positions = np.arange(component_mask.shape[1], dtype=np.uint64)[None, :]
    seed_state = np.uint64(
        (int(seed) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    )
    epoch_state = np.uint64(
        ((int(epoch) + 1) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    )
    states = (
        np.asarray(row_ids, dtype=np.uint64)[:, None]
        ^ seed_state
        ^ epoch_state
        ^ (positions * np.uint64(0x94D049BB133111EB))
    )
    uniform = _splitmix64_uniform(states)  # (B, K)
    exponential = -np.log(np.clip(uniform, np.finfo(np.float64).tiny, 1.0))
    exponential *= component_mask
    denominator = exponential.sum(axis=1, keepdims=True)
    weights = np.divide(
        exponential,
        denominator,
        out=np.zeros_like(exponential),
        where=denominator > 0,
    ).astype(np.float32)
    return torch.as_tensor(weights)  # (B, K)


def _splitmix64_uniform(values: np.ndarray) -> np.ndarray:
    """Map uint64 states to deterministic open-unit-interval values."""
    state = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    state = (state ^ (state >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    state = (state ^ (state >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    state = state ^ (state >> np.uint64(31))
    return ((state >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)


def _add_blank_profiles(
    spectra: torch.Tensor,
    centers: torch.Tensor,
    *,
    radius: int,
) -> None:
    """Add normalized triangular profiles at blank axis anchors."""
    active = centers >= 0  # (B,)
    if not bool(active.any()):
        return
    offsets = torch.arange(-radius, radius + 1, dtype=torch.long)  # (W,)
    positions = centers[:, None] + offsets[None, :]  # (B, W)
    valid = (
        active[:, None]
        & (positions >= 0)
        & (positions < spectra.shape[1])
    )  # (B, W)
    profile = 1.0 - offsets.abs().to(dtype=spectra.dtype) / float(radius + 1)  # (W,)
    values = profile.expand_as(positions).clone() * valid.to(dtype=spectra.dtype)  # (B, W)
    values = values / values.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(spectra.dtype).tiny
    )  # (B, W)
    spectra.scatter_add_(
        1,
        positions.clamp(0, spectra.shape[1] - 1),
        values,
    )


def _evenly_spaced_indices(size: int, count: int) -> np.ndarray:
    """Select deterministic validation rows across a static manifest."""
    if count < 1 or count > size:
        raise ValueError("Validation row count is outside the manifest range.")
    if count == size:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, count, dtype=np.int64)
