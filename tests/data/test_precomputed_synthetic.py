"""Contracts for persistent, batch-rendered synthetic pretraining artifacts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Subset

from msi_autoencoder_wrapper.data import TargetSchema
from msi_autoencoder_wrapper.data.pretraining.precomputed import (
    PrecomputedSyntheticConfig,
    PrecomputedSyntheticDataset,
    SyntheticArtifactStore,
    SyntheticPrecomputeBuilder,
    _component_weights,
    build_precomputed_synthetic_partitions,
)
from msi_autoencoder_wrapper.data.pretraining.annotation_population import (
    AnnotationPopulation,
)
from msi_autoencoder_wrapper.models.datasets.annotations.index import (
    MappedSpectrumAnnotationIndex,
)
from msi_autoencoder_wrapper.models.datasets.splitting.partitions import (
    DatasetPartitions,
    SplitManifest,
)
from msi_autoencoder_wrapper.data.supervision_masks import (
    simulated_negative_mask_key,
)


class TinyPrecomputeDataset:
    """Small train-only annotation fixture with grouped and blank axis bins."""

    dtype = torch.float32

    def __init__(self) -> None:
        self.active_context = SimpleNamespace(
            binner=SimpleNamespace(GetXAxis=lambda: np.arange(8, dtype=np.float64))
        )

    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int):
        return index

    def create_partitions(self) -> DatasetPartitions:
        return DatasetPartitions(
            Subset(self, [0, 1, 2, 3]),
            Subset(self, [4]),
            Subset(self, []),
            SplitManifest(
                "predefined",
                42,
                {"train": (0, 1, 2, 3), "validation": (4,), "test": ()},
            ),
        )

    @staticmethod
    def get_target_schemas() -> dict[str, TargetSchema]:
        return {
            "molecule": TargetSchema(
                "molecule",
                "multi_label",
                ("A|+H", "B|+H", "C|+H"),
            )
        }

    @staticmethod
    def get_mapped_annotation_index() -> MappedSpectrumAnnotationIndex:
        return MappedSpectrumAnnotationIndex(
            spectrum_ids=np.array([0, 1, 3], dtype=np.int64),
            spectrum_offsets=np.array([0, 2, 3, 4], dtype=np.int64),
            annotation_indices=np.array([0, 1, 1, 2], dtype=np.int64),
            coordinate_indices=np.array([1, 1, 3, 5], dtype=np.int64),
            annotation_identities=(("A", "+H"), ("B", "+H"), ("C", "+H")),
            coordinate_axis=np.arange(8, dtype=np.float64),
            coordinate_system="binner",
        )


def _parameters(cache_directory, population: str = "axis") -> dict:
    return {
        "kind": "precomputed_synthetic",
        "population": population,
        "validation_samples": 3,
        "artifact": {
            "key": "tiny-synthetic",
            "cache_directory": str(cache_directory),
            "peak_source": "annotation",
            "seed": 17,
            "normalization": "tic",
            "representation": {
                "strategy": "triangular_peak",
                "parameters": {"peak_radius": 1, "minimum_intensity": 1.0},
            },
            "blank_peak_radius": 1,
            "populations": {
                "axis": {
                    "strategy": "axis_coverage",
                    "repetitions_per_bin": 2,
                },
                "mixture": {
                    "strategy": "uniform_mixture",
                    "repetitions_per_bin": 3,
                    "min_fragments": 2,
                    "max_fragments": 2,
                    "detection_probability": 1.0,
                },
            },
        },
    }


def _artifact(tmp_path):
    dataset = TinyPrecomputeDataset()
    config = PrecomputedSyntheticConfig.from_mapping(_parameters(tmp_path))
    builder = SyntheticPrecomputeBuilder(dataset, config)
    return SyntheticArtifactStore(config.cache_directory).load_or_build(
        artifact_key=config.artifact_key,
        fingerprint=builder.fingerprint,
        builder=builder.build,
    )


def test_axis_manifest_has_exact_repetitions_and_complete_negative_targets(tmp_path):
    """Axis coverage emits every bin twice and fully known molecular classes."""
    artifact = _artifact(tmp_path)
    manifest = artifact.manifests["axis"]
    assert manifest.component_ids.shape[0] == 16
    np.testing.assert_array_equal(
        np.bincount(manifest.anchor_bins, minlength=8),
        np.full(8, 2),
    )
    dataset = PrecomputedSyntheticDataset(
        artifact,
        population="axis",
        schemas=TinyPrecomputeDataset.get_target_schemas(),
        dtype=torch.float32,
        seed=17,
    )
    bin_one_rows = np.flatnonzero(manifest.anchor_bins == 1)
    blank_rows = np.flatnonzero(manifest.anchor_bins == 0)
    batch = dataset.collate_fn([int(bin_one_rows[0]), int(blank_rows[0])])
    torch.testing.assert_close(
        batch.spectra.sum(dim=1),
        torch.ones(2),
        rtol=1e-6,
        atol=1e-7,
    )
    assert bool((batch.spectra >= 0).all())
    assert batch.targets.values["molecule"].tolist() == [
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert bool(batch.targets.masks["molecule"].all())
    assert batch.targets.masks[simulated_negative_mask_key("molecule")].tolist() == [
        [False, False, True],
        [True, True, True],
    ]


def test_uniform_mixture_meets_quota_and_uses_distinct_anchor_bins(tmp_path):
    """Every nonempty anchor bin meets quota without duplicate bin groups per row."""
    artifact = _artifact(tmp_path)
    manifest = artifact.manifests["mixture"]
    valid = manifest.component_ids >= 0
    anchors = np.where(
        valid,
        artifact.prototype_anchor_bins[
            np.clip(manifest.component_ids, 0, None)
        ],
        -1,
    )
    for anchor in (1, 3, 5):
        assert int((anchors == anchor).any(axis=1).sum()) >= 3
    for row in anchors:
        selected = row[row >= 0]
        assert 2 <= len(set(selected.tolist())) <= 2


def test_sparse_batch_renderer_matches_independent_dense_reference(tmp_path):
    """Sparse matrix rendering equals a transparent per-component dense sum."""
    artifact = _artifact(tmp_path)
    dataset = PrecomputedSyntheticDataset(
        artifact,
        population="mixture",
        schemas=TinyPrecomputeDataset.get_target_schemas(),
        dtype=torch.float32,
        seed=17,
    )
    rows = np.array([0, 1], dtype=np.int64)
    batch = dataset.collate_fn(rows.tolist())
    dense_basis = np.zeros(
        (artifact.prototype_count, artifact.feature_count),
        dtype=np.float64,
    )
    np.add.at(
        dense_basis,
        (artifact.basis_rows, artifact.basis_columns),
        artifact.basis_values,
    )
    component_ids = artifact.manifests["mixture"].component_ids[rows]
    weights = _component_weights(
        row_ids=rows,
        component_mask=component_ids >= 0,
        seed=17,
        epoch=0,
    ).numpy()
    expected = np.zeros((len(rows), artifact.feature_count), dtype=np.float64)
    for row_index, row in enumerate(component_ids):
        for component_index, prototype_id in enumerate(row):
            if prototype_id >= 0:
                expected[row_index] += (
                    weights[row_index, component_index]
                    * dense_basis[prototype_id]
                )
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(
        batch.spectra.numpy(),
        expected.astype(np.float32),
        rtol=1e-6,
        atol=1e-7,
    )


def test_epoch_changes_weights_without_changing_static_manifest(tmp_path):
    """Epoch variation affects mixture proportions but not static artifact data."""
    artifact = _artifact(tmp_path)
    dataset = PrecomputedSyntheticDataset(
        artifact,
        population="mixture",
        schemas=TinyPrecomputeDataset.get_target_schemas(),
        dtype=torch.float32,
        seed=17,
    )
    static_ids = artifact.manifests["mixture"].component_ids.copy()
    first = dataset.collate_fn([0, 1]).spectra
    dataset.set_epoch(1)
    second = dataset.collate_fn([0, 1]).spectra
    np.testing.assert_array_equal(
        artifact.manifests["mixture"].component_ids,
        static_ids,
    )
    assert not torch.equal(first, second)
    dataset.set_epoch(1)
    torch.testing.assert_close(second, dataset.collate_fn([0, 1]).spectra, rtol=0, atol=0)


def test_matching_artifact_is_loaded_without_invoking_builder(tmp_path):
    """A second matching request is a cache hit rather than another build."""
    artifact = _artifact(tmp_path)
    store = SyntheticArtifactStore(tmp_path)

    def fail_builder():
        raise AssertionError("Matching artifact unexpectedly rebuilt.")

    loaded = store.load_or_build(
        artifact_key=artifact.artifact_key,
        fingerprint=artifact.fingerprint,
        builder=fail_builder,
    )
    assert loaded.fingerprint == artifact.fingerprint
    np.testing.assert_array_equal(loaded.basis_values, artifact.basis_values)


def test_partition_cache_hit_skips_annotation_geometry_construction(
    tmp_path,
    monkeypatch,
):
    """A reused artifact does not reconstruct annotation candidates or spectra."""
    parameters = _parameters(tmp_path)
    first = build_precomputed_synthetic_partitions(
        TinyPrecomputeDataset(),
        parameters,
    )

    def fail_annotation_geometry(*_args, **_kwargs):
        raise AssertionError("Cache hit unexpectedly rebuilt annotation geometry.")

    monkeypatch.setattr(
        "msi_autoencoder_wrapper.data.pretraining.precompute_builder."
        "_compact_annotation_bins",
        fail_annotation_geometry,
    )
    second = build_precomputed_synthetic_partitions(
        TinyPrecomputeDataset(),
        parameters,
    )
    assert len(second["train"]) == len(first["train"])


def test_partition_builder_selects_fixed_validation_rows(tmp_path):
    """Validation remains deterministic while train phases change epoch weights."""
    partitions = build_precomputed_synthetic_partitions(
        TinyPrecomputeDataset(),
        _parameters(tmp_path, population="mixture"),
    )
    train = partitions["train"]
    validation = partitions["validation"]
    assert len(validation) == 3
    validation.set_epoch(9)
    first = validation.collate_fn([0, 1]).spectra
    validation.set_epoch(10)
    second = validation.collate_fn([0, 1]).spectra
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    train.set_epoch(9)
    train_spectra = train.collate_fn([0, 1]).spectra
    train.set_epoch(10)
    assert not torch.equal(train_spectra, train.collate_fn([0, 1]).spectra)


def test_sparse_annotation_extraction_does_not_query_unannotated_rows():
    """The annotation cache touches only sparse index rows present in the train split."""
    dataset = TinyPrecomputeDataset()
    index = dataset.get_mapped_annotation_index()
    proxy = SimpleNamespace(
        spectrum_ids=index.spectrum_ids,
        spectrum_offsets=index.spectrum_offsets,
        annotation_indices=index.annotation_indices,
        coordinate_indices=index.coordinate_indices,
        annotation_identities=index.annotation_identities,
        coordinate_axis=index.coordinate_axis,
        coordinate_system=index.coordinate_system,
        entry_slice=lambda _: (_ for _ in ()).throw(
            AssertionError("Per-spectrum index lookup is not allowed.")
        ),
    )
    dataset.get_mapped_annotation_index = lambda: proxy
    population = AnnotationPopulation.from_dataset(dataset)
    assert population.positive_labels == (0, 1, 2)
