"""Compile persistent sparse artifacts for synthetic spectral pretraining.

The builder stores one sparse theoretical basis and compact integer manifests.
Dense spectra are rendered only for the requested DataLoader batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from ...utils.logger import get_custom_logger
from .precompute_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    SyntheticArtifactStore,
    SyntheticManifest,
    SyntheticPrecomputeArtifact,
)
from .precompute_dataset import PrecomputedSyntheticDataset
from .representations import (
    SyntheticRepresentationContext,
    SyntheticRepresentationSpec,
    get_representation_strategy,
)
from .sampling import SyntheticComponent, SyntheticSampleDefinition
from .sources import CandidateCatalogPeakSource, SyntheticPeakSource

logger = get_custom_logger(__name__)

_EMPTY_COMPONENT = -1
_KIND_PADDING = 0
_KIND_SINGLE = 1
_KIND_BASE_CLASS = 2
_KIND_BLANK = 3
_KIND_OVERLAP_BONUS = 4
_KIND_RARE_BONUS = 5


@dataclass(frozen=True)
class PrecomputedPopulationSpec:
    """One population compiled into an immutable artifact manifest."""

    name: str
    strategy: str
    repetitions_per_bin: int | None = None
    min_fragments: int | None = None
    max_fragments: int | None = None
    detection_probability: float = 1.0
    class_quota: int = 0
    blank_bin_quota: int = 0
    overlap_bonus_per_class: int = 0
    rare_bonus_per_class: int = 0
    rare_fraction: float = 0.25
    members: tuple[str, ...] = ()

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
        supported = {
            "axis_coverage",
            "uniform_mixture",
            "class_quota_mixture",
            "combined",
        }
        if strategy not in supported:
            raise ValueError(
                f"Population strategy must be one of {sorted(supported)}."
            )

        repetitions = value.get("repetitions_per_bin")
        minimum = value.get("min_fragments")
        maximum = value.get("max_fragments")
        probability = value.get("detection_probability", 1.0)
        class_quota = value.get("class_quota", 0)
        blank_quota = value.get("blank_bin_quota", 0)
        overlap_bonus = value.get("overlap_bonus_per_class", 0)
        rare_bonus = value.get("rare_bonus_per_class", 0)
        rare_fraction = value.get("rare_fraction", 0.25)
        members_value = value.get("members", ())

        common_keys = {"strategy"}
        if strategy == "axis_coverage":
            allowed = common_keys | {"repetitions_per_bin"}
            _positive_integer(repetitions, "repetitions_per_bin")
        elif strategy == "uniform_mixture":
            allowed = common_keys | {
                "repetitions_per_bin",
                "min_fragments",
                "max_fragments",
                "detection_probability",
            }
            _positive_integer(repetitions, "repetitions_per_bin")
            _fragment_bounds(minimum, maximum, strategy)
            if (
                not isinstance(probability, (int, float))
                or not 0.0 < float(probability) <= 1.0
            ):
                raise ValueError("detection_probability must belong to (0, 1].")
        elif strategy == "class_quota_mixture":
            allowed = common_keys | {
                "class_quota",
                "blank_bin_quota",
                "overlap_bonus_per_class",
                "rare_bonus_per_class",
                "rare_fraction",
                "min_fragments",
                "max_fragments",
            }
            _fragment_bounds(minimum, maximum, strategy)
            for token, configured in (
                ("class_quota", class_quota),
                ("blank_bin_quota", blank_quota),
                ("overlap_bonus_per_class", overlap_bonus),
                ("rare_bonus_per_class", rare_bonus),
            ):
                _nonnegative_integer(configured, token)
            if int(class_quota) == 0 and int(blank_quota) == 0:
                raise ValueError(
                    "class_quota_mixture requires a class or blank-bin quota."
                )
            if (
                not isinstance(rare_fraction, (int, float))
                or not 0.0 < float(rare_fraction) <= 1.0
            ):
                raise ValueError("rare_fraction must belong to (0, 1].")
        else:
            allowed = common_keys | {"members"}
            if (
                not isinstance(members_value, Sequence)
                or isinstance(members_value, str)
                or len(members_value) < 2
                or not all(isinstance(item, str) and item for item in members_value)
            ):
                raise ValueError(
                    "combined populations require at least two named members."
                )

        unsupported = set(value).difference(allowed)
        if unsupported:
            raise ValueError(
                f"Unsupported precomputed population keys: {sorted(unsupported)}."
            )
        return cls(
            name=name,
            strategy=str(strategy),
            repetitions_per_bin=(
                int(repetitions) if repetitions is not None else None
            ),
            min_fragments=int(minimum) if minimum is not None else None,
            max_fragments=int(maximum) if maximum is not None else None,
            detection_probability=float(probability),
            class_quota=int(class_quota),
            blank_bin_quota=int(blank_quota),
            overlap_bonus_per_class=int(overlap_bonus),
            rare_bonus_per_class=int(rare_bonus),
            rare_fraction=float(rare_fraction),
            members=tuple(str(item) for item in members_value),
        )

    def fingerprint_payload(self) -> dict[str, Any]:
        """Return JSON-compatible population identity."""
        return {
            "name": self.name,
            "strategy": self.strategy,
            "repetitions_per_bin": self.repetitions_per_bin,
            "min_fragments": self.min_fragments,
            "max_fragments": self.max_fragments,
            "detection_probability": self.detection_probability,
            "class_quota": self.class_quota,
            "blank_bin_quota": self.blank_bin_quota,
            "overlap_bonus_per_class": self.overlap_bonus_per_class,
            "rare_bonus_per_class": self.rare_bonus_per_class,
            "rare_fraction": self.rare_fraction,
            "members": self.members,
        }


@dataclass(frozen=True)
class PrecomputedSyntheticConfig:
    """Immutable request for artifact-backed synthetic populations."""

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
    minimum_annotated_concentration: float
    blank_concentration: float
    candidate_filters: Mapping[str, Any] | None
    candidate_classes: tuple[str, ...] | None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "PrecomputedSyntheticConfig":
        """Parse an explicit ``precomputed_synthetic`` phase declaration."""
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
        names = {population.name for population in populations}
        if selected_population not in names:
            raise ValueError(
                "pretraining.population must select an artifact population."
            )
        for population in populations:
            missing = set(population.members).difference(names)
            if missing:
                raise ValueError(
                    f"Population '{population.name}' has unknown members: "
                    f"{sorted(missing)}."
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
        _positive_integer(validation_samples, "pretraining.validation_samples")
        blank_peak_radius = artifact.get("blank_peak_radius", 0)
        _nonnegative_integer(
            blank_peak_radius,
            "pretraining.artifact.blank_peak_radius",
        )

        mixing = artifact.get("component_mixing", {})
        if not isinstance(mixing, Mapping):
            raise ValueError("pretraining.artifact.component_mixing must be a mapping.")
        unsupported_mixing = set(mixing).difference(
            {
                "distribution",
                "minimum_annotated_concentration",
                "blank_concentration",
            }
        )
        if unsupported_mixing:
            raise ValueError(
                f"Unsupported component_mixing keys: {sorted(unsupported_mixing)}."
            )
        if mixing.get("distribution", "dirichlet") != "dirichlet":
            raise ValueError("component_mixing.distribution must be 'dirichlet'.")
        minimum_annotated = mixing.get("minimum_annotated_concentration", 2.0)
        blank_concentration = mixing.get("blank_concentration", 1.0)
        for token, configured in (
            ("minimum_annotated_concentration", minimum_annotated),
            ("blank_concentration", blank_concentration),
        ):
            if (
                not isinstance(configured, (int, float))
                or not np.isfinite(configured)
                or float(configured) <= 0
            ):
                raise ValueError(f"component_mixing.{token} must be positive and finite.")

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
            validation_samples=int(validation_samples),
            blank_peak_radius=int(blank_peak_radius),
            minimum_annotated_concentration=float(minimum_annotated),
            blank_concentration=float(blank_concentration),
            candidate_filters=(
                dict(candidate_filters) if candidate_filters is not None else None
            ),
            candidate_classes=(
                tuple(candidate_classes) if candidate_classes is not None else None
            ),
        )


@dataclass(frozen=True)
class _AnnotationSummary:
    """Compact train-only annotation geometry and class-frequency statistics."""

    bins: tuple[tuple[int, ...], ...]
    class_counts: tuple[int, ...]
    overlap_target_indices: tuple[int, ...]


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
    """Compile a train-only source, sparse theoretical basis, and manifests."""

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
            self.class_counts,
            self.overlap_targets,
        ) = self._build_source()
        (
            self.prototype_labels,
            self.prototype_centers,
            self.prototype_source_labels,
            self.anchor_groups,
            self.class_prototypes,
        ) = self._build_prototypes()
        self.blank_bins = np.asarray(
            [index for index, group in enumerate(self.anchor_groups) if not group],
            dtype=np.int32,
        )
        self.eligible_targets = tuple(
            index for index, prototypes in enumerate(self.class_prototypes) if prototypes
        )
        self.fingerprint = fingerprint or self._fingerprint()

    def build(self) -> SyntheticPrecomputeArtifact:
        """Render each theoretical profile once and compile all manifests."""
        basis_rows, basis_columns, basis_values = self._build_sparse_basis()
        specifications = {population.name: population for population in self.config.populations}
        manifests: dict[str, SyntheticManifest] = {}
        active: set[str] = set()

        def compile_named(name: str) -> SyntheticManifest:
            if name in manifests:
                return manifests[name]
            if name in active:
                raise ValueError("Combined population declarations contain a cycle.")
            active.add(name)
            population = specifications[name]
            if population.strategy == "combined":
                members = [compile_named(member) for member in population.members]
                manifest = self._combine_manifests(population, members)
            else:
                manifest = self._compile_manifest(population)
            active.remove(name)
            manifests[name] = manifest
            return manifest

        ordered_manifests = {
            population.name: compile_named(population.name)
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
            manifests=ordered_manifests,
            blank_peak_radius=self.config.blank_peak_radius,
            normalization=self.config.normalization,
        )

    def _get_context(self) -> Any:
        """Resolve binner and optional candidate-catalogue context."""
        getter = getattr(self.dataset, "get_synthetic_context", None)
        return getter() if callable(getter) else self.dataset.active_context

    def _build_source(
        self,
    ) -> tuple[
        SyntheticPeakSource,
        np.ndarray,
        Mapping[str, Any],
        np.ndarray,
        tuple[int, ...],
    ]:
        """Resolve source geometry and compact train-only statistics."""
        target_names = self.schemas["molecule"].class_names
        target_lookup = {name: index for index, name in enumerate(target_names)}
        if self.config.peak_source == "annotation":
            index = self.dataset.get_mapped_annotation_index()
            train_ids = _partition_source_ids(self.dataset.create_partitions().train)
            summary = _compact_annotation_summary(
                index=index,
                selected_spectrum_ids=train_ids,
                target_indices=target_lookup,
                target_count=len(target_names),
                feature_count=self.axis.size,
            )
            source = _StaticPeakSource(
                feature_count=self.axis.size,
                class_names=tuple(target_names),
                bins=summary.bins,
            )
            signature = {
                "kind": "annotation",
                "spectrum_ids": tuple(sorted(set(int(value) for value in train_ids))),
                "bins": summary.bins,
                "class_counts": summary.class_counts,
                "overlap_target_indices": summary.overlap_target_indices,
            }
            return (
                source,
                np.arange(len(target_names), dtype=np.int32),
                signature,
                np.asarray(summary.class_counts, dtype=np.int64),
                summary.overlap_target_indices,
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
        class_counts = np.zeros(len(target_names), dtype=np.int64)
        for source_label, target_label in enumerate(source_to_target):
            class_counts[int(target_label)] += len(source.bins[source_label])
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
        return source, source_to_target, signature, class_counts, ()

    def _build_prototypes(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[tuple[int, ...], ...],
        tuple[tuple[int, ...], ...],
    ]:
        """Create reusable profiles and their axis/class lookup tables."""
        labels: list[int] = []
        centers: list[int] = []
        source_labels: list[int] = []
        groups: list[list[int]] = [[] for _ in range(self.axis.size)]
        by_target: list[list[int]] = [
            [] for _ in range(self.schemas["molecule"].class_count)
        ]
        one_profile_per_class = self.config.representation.strategy == "isospec_envelope"
        for source_label, raw_coordinates in enumerate(self.source.bins):
            target_label = int(self.source_to_target[source_label])
            coordinates = tuple(sorted(set(int(value) for value in raw_coordinates)))
            if target_label < 0 or not coordinates:
                continue
            if any(
                coordinate < 0 or coordinate >= self.axis.size
                for coordinate in coordinates
            ):
                raise ValueError(
                    "Synthetic source coordinate is outside the active axis."
                )
            if one_profile_per_class:
                prototype_id = len(labels)
                labels.append(target_label)
                centers.append(coordinates[0])
                source_labels.append(source_label)
                by_target[target_label].append(prototype_id)
                for coordinate in coordinates:
                    groups[coordinate].append(prototype_id)
                continue
            for coordinate in coordinates:
                prototype_id = len(labels)
                labels.append(target_label)
                centers.append(coordinate)
                source_labels.append(source_label)
                by_target[target_label].append(prototype_id)
                groups[coordinate].append(prototype_id)
        return (
            np.asarray(labels, dtype=np.int32),
            np.asarray(centers, dtype=np.int32),
            np.asarray(source_labels, dtype=np.int32),
            tuple(tuple(group) for group in groups),
            tuple(tuple(group) for group in by_target),
        )

    def _fingerprint(self) -> str:
        """Return content identity for immutable artifact inputs."""
        digest = sha256()
        digest.update(
            json.dumps(
                _fingerprint_payload(self.config, self.schemas),
                sort_keys=True,
                default=list,
            ).encode("utf-8")
        )
        digest.update(
            json.dumps(self.source_signature, sort_keys=True, default=list).encode(
                "utf-8"
            )
        )
        digest.update(self.axis.tobytes())
        digest.update(self.prototype_labels.tobytes())
        digest.update(self.prototype_centers.tobytes())
        return digest.hexdigest()

    def _build_sparse_basis(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render every immutable prototype once and retain nonzero bins."""
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
        for prototype_id, (source_label, center) in enumerate(
            zip(
                self.prototype_source_labels,
                self.prototype_centers,
                strict=True,
            )
        ):
            definition = SyntheticSampleDefinition(
                components=(
                    SyntheticComponent(
                        center=int(center),
                        label_index=int(source_label),
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
            rows.append(np.full(nonzero.size, prototype_id, dtype=np.int32))
            columns.append(nonzero.astype(np.int32, copy=False))
            values.append(spectrum[nonzero].astype(np.float32, copy=False))
        if not rows:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )
        return np.concatenate(rows), np.concatenate(columns), np.concatenate(values)

    def _compile_manifest(
        self,
        population: PrecomputedPopulationSpec,
    ) -> SyntheticManifest:
        """Compile one non-combined population."""
        if population.strategy == "axis_coverage":
            return self._compile_axis_manifest(population)
        if population.strategy == "uniform_mixture":
            return self._compile_uniform_mixture_manifest(population)
        return self._compile_class_quota_manifest(population)

    def _compile_axis_manifest(
        self,
        population: PrecomputedPopulationSpec,
    ) -> SyntheticManifest:
        """Compile exact configurable repetitions of every active-axis bin."""
        anchors = np.tile(
            np.arange(self.axis.size, dtype=np.int32),
            int(population.repetitions_per_bin),
        )
        component_rows: list[tuple[int, ...]] = []
        blank_rows: list[tuple[int, ...]] = []
        target_rows: list[tuple[int, ...]] = []
        kind_rows: list[tuple[int, ...]] = []
        for anchor in anchors:
            components = self.anchor_groups[int(anchor)]
            if components:
                component_rows.append(components)
                blank_rows.append(tuple(-1 for _ in components))
                target_rows.append(
                    tuple(int(self.prototype_labels[item]) for item in components)
                )
                kind_rows.append(tuple(_KIND_SINGLE for _ in components))
            else:
                component_rows.append((_EMPTY_COMPONENT,))
                blank_rows.append((int(anchor),))
                target_rows.append((_EMPTY_COMPONENT,))
                kind_rows.append((_KIND_SINGLE,))
        size = len(component_rows)
        logger.info(
            "Materialized artifact population '%s' (not a training phase): "
            "axis_bins=%s repetitions_per_bin=%s samples=%s.",
            population.name,
            self.axis.size,
            population.repetitions_per_bin,
            size,
        )
        return _make_manifest(
            component_rows=component_rows,
            blank_rows=blank_rows,
            target_rows=target_rows,
            kind_rows=kind_rows,
            anchors=anchors,
            annotated_concentrations=np.full(
                size,
                self.config.minimum_annotated_concentration,
                dtype=np.float32,
            ),
            blank_concentrations=np.full(
                size,
                self.config.blank_concentration,
                dtype=np.float32,
            ),
        )

    def _compile_uniform_mixture_manifest(
        self,
        population: PrecomputedPopulationSpec,
    ) -> SyntheticManifest:
        """Compile the backward-compatible nonempty-bin mixture strategy."""
        eligible_bins = np.asarray(
            [index for index, group in enumerate(self.anchor_groups) if group],
            dtype=np.int32,
        )
        if eligible_bins.size < int(population.min_fragments):
            raise ValueError(
                "uniform_mixture requires at least min_fragments nonempty bins."
            )
        rng = self._population_rng(population.name)
        remaining = np.full(
            eligible_bins.size,
            int(population.repetitions_per_bin),
            dtype=np.int32,
        )
        all_positions = np.arange(eligible_bins.size, dtype=np.int32)
        component_rows: list[tuple[int, ...]] = []
        while bool(np.any(remaining > 0)):
            requested = int(
                rng.integers(
                    int(population.min_fragments),
                    int(population.max_fragments) + 1,
                )
            )
            active = all_positions[remaining > 0]
            priority = np.lexsort((rng.random(active.size), -remaining[active]))
            selected = active[priority[: min(requested, active.size)]]
            if selected.size < requested:
                available = np.setdiff1d(all_positions, selected, assume_unique=True)
                selected = np.concatenate(
                    (
                        selected,
                        rng.choice(
                            available,
                            size=requested - selected.size,
                            replace=False,
                        ),
                    )
                )
            retained: list[int] = []
            for anchor in eligible_bins[selected]:
                group = self.anchor_groups[int(anchor)]
                keep = rng.random(len(group)) < population.detection_probability
                if not bool(keep.any()):
                    keep[int(rng.integers(len(group)))] = True
                retained.extend(
                    prototype
                    for prototype, selected_component in zip(group, keep, strict=True)
                    if selected_component
                )
            component_rows.append(tuple(retained))
            selected_active = selected[remaining[selected] > 0]
            remaining[selected_active] -= 1
        target_rows = [
            tuple(int(self.prototype_labels[item]) for item in row)
            for row in component_rows
        ]
        size = len(component_rows)
        return _make_manifest(
            component_rows=component_rows,
            blank_rows=[tuple(-1 for _ in row) for row in component_rows],
            target_rows=target_rows,
            kind_rows=[tuple(_KIND_BASE_CLASS for _ in row) for row in component_rows],
            anchors=np.asarray(
                [int(self.prototype_centers[row[0]]) for row in component_rows],
                dtype=np.int32,
            ),
            annotated_concentrations=np.full(
                size,
                self.config.minimum_annotated_concentration,
                dtype=np.float32,
            ),
            blank_concentrations=np.full(
                size,
                self.config.blank_concentration,
                dtype=np.float32,
            ),
        )

    def _compile_class_quota_manifest(
        self,
        population: PrecomputedPopulationSpec,
    ) -> SyntheticManifest:
        """Compile exact class and empty-bin quotas with additive bonuses."""
        if population.class_quota and not self.eligible_targets:
            raise ValueError("Class quota requires at least one annotated class.")
        rng = self._population_rng(population.name)
        tokens: list[tuple[int, int, int, int]] = []

        # Base quota
        ## Annotated tokens request classes; their theoretical profile is
        ## sampled independently from the class-specific sparse basis bank.
        for target in self.eligible_targets:
            for _ in range(population.class_quota):
                tokens.append(self._class_token(rng, target, _KIND_BASE_CLASS))
        for center in self.blank_bins:
            tokens.extend(
                (
                    _EMPTY_COMPONENT,
                    int(center),
                    _EMPTY_COMPONENT,
                    _KIND_BLANK,
                )
                for _ in range(population.blank_bin_quota)
            )

        # Additive enrichment quotas
        ## A class in both sets receives both independent bonuses.
        eligible = set(self.eligible_targets)
        for target in sorted(eligible.intersection(self.overlap_targets)):
            for _ in range(population.overlap_bonus_per_class):
                tokens.append(self._class_token(rng, target, _KIND_OVERLAP_BONUS))
        for target in self._rare_targets(population.rare_fraction):
            for _ in range(population.rare_bonus_per_class):
                tokens.append(self._class_token(rng, target, _KIND_RARE_BONUS))
        if not tokens:
            raise ValueError("Class-quota mixture produced no component tokens.")
        if len(tokens) < int(population.min_fragments):
            raise ValueError(
                "Class-quota token count is smaller than min_fragments."
            )
        order = rng.permutation(len(tokens))
        shuffled = [tokens[int(index)] for index in order]
        sizes = _random_partition_sizes(
            len(shuffled),
            int(population.min_fragments),
            int(population.max_fragments),
            rng,
        )
        component_rows: list[tuple[int, ...]] = []
        blank_rows: list[tuple[int, ...]] = []
        target_rows: list[tuple[int, ...]] = []
        kind_rows: list[tuple[int, ...]] = []
        anchors: list[int] = []
        cursor = 0
        for size in sizes:
            row = shuffled[cursor : cursor + size]
            cursor += size
            component_rows.append(tuple(item[0] for item in row))
            blank_rows.append(tuple(item[1] for item in row))
            target_rows.append(tuple(item[2] for item in row))
            kind_rows.append(tuple(item[3] for item in row))
            first = row[0]
            anchors.append(
                int(self.prototype_centers[first[0]])
                if first[0] >= 0
                else int(first[1])
            )

        # Population-calibrated intensity prior
        ## REMARK: Setting alpha_annotated / alpha_blank to the inverse token
        ## frequency ratio gives both groups equal expected aggregate mass when
        ## a row follows the population composition. The configured minimum
        ## preserves a per-component preference for annotated spectra when the
        ## additive overlap/rare quotas make annotations more frequent.
        annotated_count = sum(item[0] >= 0 for item in tokens)
        blank_count = len(tokens) - annotated_count
        annotated_concentration = self.config.minimum_annotated_concentration
        if annotated_count and blank_count:
            annotated_concentration = max(
                annotated_concentration,
                self.config.blank_concentration * blank_count / annotated_count,
            )
        logger.info(
            "Materialized artifact population '%s' (not a training phase): "
            "annotated=%s blank=%s "
            "alpha_annotated=%s alpha_blank=%s samples=%s.",
            population.name,
            annotated_count,
            blank_count,
            annotated_concentration,
            self.config.blank_concentration,
            len(component_rows),
        )
        sample_count = len(component_rows)
        return _make_manifest(
            component_rows=component_rows,
            blank_rows=blank_rows,
            target_rows=target_rows,
            kind_rows=kind_rows,
            anchors=np.asarray(anchors, dtype=np.int32),
            annotated_concentrations=np.full(
                sample_count,
                annotated_concentration,
                dtype=np.float32,
            ),
            blank_concentrations=np.full(
                sample_count,
                self.config.blank_concentration,
                dtype=np.float32,
            ),
        )

    def _class_token(
        self,
        rng: np.random.Generator,
        target: int,
        kind: int,
    ) -> tuple[int, int, int, int]:
        """Sample one theoretical profile while retaining requested class."""
        prototypes = self.class_prototypes[int(target)]
        prototype = prototypes[int(rng.integers(len(prototypes)))]
        return int(prototype), _EMPTY_COMPONENT, int(target), kind

    def _rare_targets(self, fraction: float) -> tuple[int, ...]:
        """Return the least frequent eligible classes under a stable cutoff."""
        count = max(1, int(np.ceil(len(self.eligible_targets) * fraction)))
        return tuple(
            sorted(
                self.eligible_targets,
                key=lambda target: (int(self.class_counts[target]), target),
            )[:count]
        )

    def _combine_manifests(
        self,
        population: PrecomputedPopulationSpec,
        members: Sequence[SyntheticManifest],
    ) -> SyntheticManifest:
        """Shuffle complete member rows into one joint training population."""
        width = max(manifest.component_ids.shape[1] for manifest in members)
        arrays: dict[str, list[np.ndarray]] = {
            "component_ids": [],
            "blank_centers": [],
            "requested_target_indices": [],
            "component_kinds": [],
        }
        for manifest in members:
            arrays["component_ids"].append(
                _pad_array(manifest.component_ids, width, _EMPTY_COMPONENT)
            )
            arrays["blank_centers"].append(
                _pad_array(manifest.blank_centers, width, _EMPTY_COMPONENT)
            )
            arrays["requested_target_indices"].append(
                _pad_array(
                    manifest.requested_target_indices,
                    width,
                    _EMPTY_COMPONENT,
                )
            )
            arrays["component_kinds"].append(
                _pad_array(manifest.component_kinds, width, _KIND_PADDING)
            )
        component_ids = np.concatenate(arrays["component_ids"], axis=0)
        blank_centers = np.concatenate(arrays["blank_centers"], axis=0)
        requested = np.concatenate(arrays["requested_target_indices"], axis=0)
        kinds = np.concatenate(arrays["component_kinds"], axis=0)
        anchors = np.concatenate([manifest.anchor_bins for manifest in members])
        annotated_concentrations = np.concatenate(
            [manifest.annotated_concentrations for manifest in members]
        )
        blank_concentrations = np.concatenate(
            [manifest.blank_concentrations for manifest in members]
        )
        order = self._population_rng(population.name).permutation(len(anchors))
        logger.info(
            "Materialized combined artifact population '%s' from members=%s: "
            "rows=%s, shuffled_together=true (not a training phase).",
            population.name,
            list(population.members),
            len(anchors),
        )
        return SyntheticManifest(
            component_ids=component_ids[order],
            blank_centers=blank_centers[order],
            requested_target_indices=requested[order],
            component_kinds=kinds[order],
            anchor_bins=anchors[order],
            annotated_concentrations=annotated_concentrations[order],
            blank_concentrations=blank_concentrations[order],
        )

    def _population_rng(self, name: str) -> np.random.Generator:
        """Return deterministic independent entropy for one population."""
        return np.random.default_rng(
            np.random.SeedSequence([self.config.seed, _stable_name_seed(name)])
        )


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
    manifest_size = artifact.manifests[config.selected_population].component_ids.shape[0]
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
    """Return cache identity without constructing theoretical profiles."""
    context_getter = getattr(dataset, "get_synthetic_context", None)
    context = context_getter() if callable(context_getter) else dataset.active_context
    axis = np.asarray(context.binner.GetXAxis(), dtype=np.float64)
    schemas = dataset.get_target_schemas()
    digest = sha256()
    payload = _fingerprint_payload(config, schemas)
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
    digest.update(json.dumps(payload, sort_keys=True, default=list).encode("utf-8"))
    digest.update(axis.tobytes())
    return digest.hexdigest()


def _fingerprint_payload(
    config: PrecomputedSyntheticConfig,
    schemas: Mapping[str, Any],
) -> dict[str, Any]:
    """Return common JSON payload for artifact cache identity."""
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_key": config.artifact_key,
        "peak_source": config.peak_source,
        "representation": {
            "strategy": config.representation.strategy,
            "parameters": config.representation.parameters,
        },
        "normalization": config.normalization,
        "blank_peak_radius": config.blank_peak_radius,
        "minimum_annotated_concentration": config.minimum_annotated_concentration,
        "blank_concentration": config.blank_concentration,
        "seed": config.seed,
        "populations": [
            population.fingerprint_payload() for population in config.populations
        ],
        "target_names": schemas["molecule"].class_names,
    }


def _compact_annotation_summary(
    *,
    index: Any,
    selected_spectrum_ids: Sequence[int],
    target_indices: Mapping[str, int],
    target_count: int,
    feature_count: int,
) -> _AnnotationSummary:
    """Extract unique class/bin geometry and compact overlap statistics.

    Runtime is linear in sparse annotation entries selected from the train
    split. Memory is ``O(C * B_c)`` for unique class/bin pairs rather than one
    Python object per source annotation.
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
    counts = np.zeros(target_count, dtype=np.int64)
    overlap: set[int] = set()
    for row_id in matched:
        start = int(index.spectrum_offsets[row_id])
        stop = int(index.spectrum_offsets[row_id + 1])
        targets = identity_targets[index.annotation_indices[start:stop]]
        coordinates = np.asarray(index.coordinate_indices[start:stop], dtype=np.int64)
        keep = (targets >= 0) & (coordinates >= 0) & (coordinates < feature_count)
        targets = targets[keep]
        coordinates = coordinates[keep]
        if not coordinates.size:
            continue
        for coordinate in np.unique(coordinates):
            labels = np.unique(targets[coordinates == coordinate])
            for target in labels:
                bins[int(target)].add(int(coordinate))
                counts[int(target)] += 1
            if labels.size > 1:
                overlap.update(int(target) for target in labels)
    return _AnnotationSummary(
        bins=tuple(tuple(sorted(values)) for values in bins),
        class_counts=tuple(int(value) for value in counts),
        overlap_target_indices=tuple(sorted(overlap)),
    )


def _make_manifest(
    *,
    component_rows: Sequence[Sequence[int]],
    blank_rows: Sequence[Sequence[int]],
    target_rows: Sequence[Sequence[int]],
    kind_rows: Sequence[Sequence[int]],
    anchors: np.ndarray,
    annotated_concentrations: np.ndarray,
    blank_concentrations: np.ndarray,
) -> SyntheticManifest:
    """Pad slot-aligned rows and construct a validated manifest."""
    widths = {
        len(component_rows),
        len(blank_rows),
        len(target_rows),
        len(kind_rows),
    }
    if len(widths) != 1:
        raise ValueError("Manifest row collections must have equal length.")
    width = max(
        1,
        max(
            (
                len(row)
                for rows in (component_rows, blank_rows, target_rows, kind_rows)
                for row in rows
            ),
            default=0,
        ),
    )
    return SyntheticManifest(
        component_ids=_pad_rows(component_rows, width, _EMPTY_COMPONENT, np.int32),
        blank_centers=_pad_rows(blank_rows, width, _EMPTY_COMPONENT, np.int32),
        requested_target_indices=_pad_rows(
            target_rows,
            width,
            _EMPTY_COMPONENT,
            np.int32,
        ),
        component_kinds=_pad_rows(
            kind_rows,
            width,
            _KIND_PADDING,
            np.uint8,
        ),
        anchor_bins=np.asarray(anchors, dtype=np.int32),
        annotated_concentrations=np.asarray(
            annotated_concentrations,
            dtype=np.float32,
        ),
        blank_concentrations=np.asarray(blank_concentrations, dtype=np.float32),
    )


def _pad_rows(
    rows: Sequence[Sequence[int]],
    width: int,
    fill: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """Pad variable-length integer rows to one dense compact manifest array."""
    result = np.full((len(rows), width), fill, dtype=dtype)
    for row_index, row in enumerate(rows):
        result[row_index, : len(row)] = row
    return result


def _pad_array(values: np.ndarray, width: int, fill: int) -> np.ndarray:
    """Right-pad one two-dimensional manifest array."""
    if values.shape[1] == width:
        return values
    result = np.full((values.shape[0], width), fill, dtype=values.dtype)
    result[:, : values.shape[1]] = values
    return result


def _random_partition_sizes(
    total: int,
    minimum: int,
    maximum: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Partition an exact token total into random feasible inclusive sizes."""
    if not _can_partition(total, minimum, maximum):
        raise ValueError(
            "Token count cannot be partitioned within min_fragments/max_fragments."
        )
    remaining = total
    sizes: list[int] = []
    while remaining:
        candidates = [
            size
            for size in range(minimum, min(maximum, remaining) + 1)
            if _can_partition(remaining - size, minimum, maximum)
        ]
        selected = int(candidates[int(rng.integers(len(candidates)))])
        sizes.append(selected)
        remaining -= selected
    return tuple(sizes)


def _can_partition(total: int, minimum: int, maximum: int) -> bool:
    """Return whether ``total`` is a sum of values in ``[minimum, maximum]``."""
    if total == 0:
        return True
    if total < minimum:
        return False
    minimum_parts = int(np.ceil(total / maximum))
    maximum_parts = total // minimum
    return minimum_parts <= maximum_parts


def _partition_source_ids(partition: Dataset) -> tuple[int, ...]:
    """Return stable source identifiers represented by a dataset partition."""
    if isinstance(partition, Subset):
        parent = _partition_source_ids(partition.dataset)
        return tuple(parent[int(index)] for index in partition.indices)
    source_ids_getter = getattr(partition, "get_sample_ids", None)
    if callable(source_ids_getter):
        return tuple(int(value) for value in source_ids_getter())
    return tuple(range(len(partition)))


def _evenly_spaced_indices(size: int, count: int) -> np.ndarray:
    """Select deterministic validation rows across a static manifest."""
    if count < 1 or count > size:
        raise ValueError("Validation row count is outside the manifest range.")
    if count == size:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, count, dtype=np.int64)


def _fragment_bounds(minimum: Any, maximum: Any, strategy: str) -> None:
    """Validate inclusive manual mixture-size bounds."""
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or minimum < 1
        or maximum < minimum
    ):
        raise ValueError(
            f"{strategy} requires 1 <= min_fragments <= max_fragments."
        )


def _positive_integer(value: Any, name: str) -> None:
    """Validate one positive integer configuration value."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _nonnegative_integer(value: Any, name: str) -> None:
    """Validate one nonnegative integer configuration value."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


def _stable_name_seed(value: str) -> int:
    """Return stable entropy for one population name."""
    return int.from_bytes(sha256(value.encode("utf-8")).digest()[:4], "big")
