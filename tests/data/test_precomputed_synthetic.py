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


class BlankHeavyPrecomputeDataset(TinyPrecomputeDataset):
    """Axis fixture whose blank-bin population exceeds its class-token count."""

    def __init__(self) -> None:
        self.active_context = SimpleNamespace(
            binner=SimpleNamespace(GetXAxis=lambda: np.arange(20, dtype=np.float64))
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
                "quota": {
                    "strategy": "class_quota_mixture",
                    "class_quota": 3,
                    "blank_bin_quota": 3,
                    "overlap_bonus_per_class": 2,
                    "rare_bonus_per_class": 2,
                    "rare_fraction": 0.34,
                    "min_fragments": 2,
                    "max_fragments": 3,
                },
                "joint": {
                    "strategy": "combined",
                    "members": ["axis", "quota"],
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
    assert batch.spectra.dtype == torch.float32
    assert batch.space.mass_axis.dtype == torch.float32
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
        annotated_mask=component_ids >= 0,
        blank_mask=np.zeros_like(component_ids, dtype=bool),
        annotated_concentrations=artifact.manifests[
            "mixture"
        ].annotated_concentrations[rows],
        blank_concentrations=artifact.manifests["mixture"].blank_concentrations[
            rows
        ],
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
        "_compact_annotation_summary",
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


def test_class_quota_manifest_has_exact_base_blank_and_additive_bonus_counts(
    tmp_path,
):
    """Requested classes and blank bins satisfy every configured exact quota."""
    manifest = _artifact(tmp_path).manifests["quota"]
    requested = manifest.requested_target_indices
    kinds = manifest.component_kinds
    blank = manifest.blank_centers
    valid_slots = (manifest.component_ids >= 0) | (blank >= 0)
    row_sizes = valid_slots.sum(axis=1)
    assert bool(((row_sizes >= 2) & (row_sizes <= 3)).all())

    # Base=3 per class; overlap A/B adds 2; rare A/C adds 2.
    np.testing.assert_array_equal(
        np.bincount(requested[requested >= 0], minlength=3),
        np.array([7, 5, 5]),
    )
    assert int((kinds == 2).sum()) == 9
    assert int((kinds == 4).sum()) == 4
    assert int((kinds == 5).sum()) == 4
    for center in (0, 2, 4, 6, 7):
        assert int((blank == center).sum()) == 3
    assert int((kinds == 3).sum()) == 15


def test_joint_population_is_a_seeded_shuffle_of_single_and_quota_rows(tmp_path):
    """Joint and staged variants differ by phase boundaries, not sample content."""
    artifact = _artifact(tmp_path)
    axis = artifact.manifests["axis"]
    quota = artifact.manifests["quota"]
    joint = artifact.manifests["joint"]
    assert joint.component_ids.shape[0] == (
        axis.component_ids.shape[0] + quota.component_ids.shape[0]
    )
    joint_width = joint.component_ids.shape[1]

    def padded(manifest):
        width = manifest.component_ids.shape[1]
        if width == joint_width:
            return manifest
        pad = joint_width - width
        return SimpleNamespace(
            component_ids=np.pad(
                manifest.component_ids,
                ((0, 0), (0, pad)),
                constant_values=-1,
            ),
            blank_centers=np.pad(
                manifest.blank_centers,
                ((0, 0), (0, pad)),
                constant_values=-1,
            ),
            requested_target_indices=np.pad(
                manifest.requested_target_indices,
                ((0, 0), (0, pad)),
                constant_values=-1,
            ),
            component_kinds=np.pad(
                manifest.component_kinds,
                ((0, 0), (0, pad)),
                constant_values=0,
            ),
            annotated_concentrations=manifest.annotated_concentrations,
            blank_concentrations=manifest.blank_concentrations,
        )

    def signatures(manifest):
        return sorted(
            (
                tuple(component.tolist()),
                tuple(blank.tolist()),
                tuple(target.tolist()),
                tuple(kinds.tolist()),
                float(annotation_alpha),
                float(blank_alpha),
            )
            for component, blank, target, kinds, annotation_alpha, blank_alpha in zip(
                manifest.component_ids,
                manifest.blank_centers,
                manifest.requested_target_indices,
                manifest.component_kinds,
                manifest.annotated_concentrations,
                manifest.blank_concentrations,
                strict=True,
            )
        )

    expected = signatures(padded(axis)) + signatures(padded(quota))
    assert signatures(joint) == sorted(expected)
    assert not np.array_equal(
        joint.component_kinds,
        np.concatenate(
            (padded(axis).component_kinds, padded(quota).component_kinds),
            axis=0,
        ),
    )


def test_weighted_dirichlet_is_reproducible_and_balances_expected_group_mass():
    """A 4:1 per-component prior balances one annotation against four blanks."""
    row_ids = np.arange(4000, dtype=np.int64)
    annotated = np.zeros((len(row_ids), 5), dtype=bool)
    annotated[:, 0] = True
    blank = ~annotated
    annotation_alpha = np.full(len(row_ids), 4.0)
    blank_alpha = np.ones(len(row_ids))
    first = _component_weights(
        row_ids=row_ids,
        annotated_mask=annotated,
        blank_mask=blank,
        annotated_concentrations=annotation_alpha,
        blank_concentrations=blank_alpha,
        seed=17,
        epoch=2,
    )
    repeated = _component_weights(
        row_ids=row_ids,
        annotated_mask=annotated,
        blank_mask=blank,
        annotated_concentrations=annotation_alpha,
        blank_concentrations=blank_alpha,
        seed=17,
        epoch=2,
    )
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    torch.testing.assert_close(first.sum(dim=1), torch.ones(len(row_ids)))
    assert float(first[:, 0].mean()) == pytest.approx(0.5, abs=0.015)


def test_population_ratio_sets_annotated_dirichlet_concentration(tmp_path):
    """Blank/class token counts set the prior ratio while preserving its floor."""
    config = PrecomputedSyntheticConfig.from_mapping(
        _parameters(tmp_path, population="quota")
    )
    artifact = SyntheticPrecomputeBuilder(
        BlankHeavyPrecomputeDataset(),
        config,
    ).build()
    manifest = artifact.manifests["quota"]

    # 17 annotated tokens: 9 base + 4 overlap + 4 rare.
    # 51 blank tokens: 17 blank bins x quota 3. Alpha 3:1 balances the
    # expected aggregate annotated and blank Dirichlet mass.
    np.testing.assert_allclose(manifest.annotated_concentrations, 3.0)
    np.testing.assert_allclose(manifest.blank_concentrations, 1.0)


def test_mixed_blank_and_annotated_batch_matches_dense_reference(tmp_path):
    """Sparse molecular profiles and blank-bin peaks use the same sampled weights."""
    artifact = _artifact(tmp_path)
    manifest = artifact.manifests["quota"]
    mixed_rows = np.flatnonzero(
        (manifest.component_ids >= 0).any(axis=1)
        & (manifest.blank_centers >= 0).any(axis=1)
    )
    assert mixed_rows.size
    row_id = int(mixed_rows[0])
    dataset = PrecomputedSyntheticDataset(
        artifact,
        population="quota",
        schemas=TinyPrecomputeDataset.get_target_schemas(),
        dtype=torch.float32,
        seed=17,
    )
    observed = dataset.collate_fn([row_id]).spectra[0].numpy()

    component_ids = manifest.component_ids[[row_id]]
    blank_centers = manifest.blank_centers[[row_id]]
    weights = _component_weights(
        row_ids=np.array([row_id]),
        annotated_mask=component_ids >= 0,
        blank_mask=blank_centers >= 0,
        annotated_concentrations=manifest.annotated_concentrations[[row_id]],
        blank_concentrations=manifest.blank_concentrations[[row_id]],
        seed=17,
        epoch=0,
    ).numpy()[0]
    expected = np.zeros(artifact.feature_count, dtype=np.float64)
    for slot, prototype_id in enumerate(component_ids[0]):
        if prototype_id < 0:
            continue
        selected = artifact.basis_rows == prototype_id
        expected[artifact.basis_columns[selected]] += (
            weights[slot] * artifact.basis_values[selected]
        )
    for slot, center in enumerate(blank_centers[0]):
        if center < 0:
            continue
        positions = np.arange(center - 1, center + 2)
        valid = (positions >= 0) & (positions < artifact.feature_count)
        profile = np.array([0.5, 1.0, 0.5], dtype=np.float64)[valid]
        profile /= profile.sum()
        expected[positions[valid]] += weights[slot] * profile
    expected /= expected.sum()
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)


def test_rendering_is_invariant_to_batch_order_and_composition(tmp_path):
    """A manifest row has the same epoch sample regardless of its batch neighbours."""
    artifact = _artifact(tmp_path)
    dataset = PrecomputedSyntheticDataset(
        artifact,
        population="quota",
        schemas=TinyPrecomputeDataset.get_target_schemas(),
        dtype=torch.float32,
        seed=17,
    )
    alone = dataset.collate_fn([1]).spectra[0]
    reordered = dataset.collate_fn([3, 1, 0]).spectra[1]
    torch.testing.assert_close(alone, reordered, rtol=0, atol=0)
