"""Tests for sweep prediction and latent-geometry evaluation records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from msi_autoencoder_wrapper.analysis.autoencoder.experiments.penalty_sweep import (
    PenaltySweepCell,
)
from msi_autoencoder_wrapper.analysis.autoencoder.experiments.sweep_evaluation import (
    ALL_CLASSES_SCOPE,
    FAIR_SCOPE,
    MaterializedSplit,
    fair_scope_mask,
    latent_geometry_frame,
    link_campaign_scratch_workspace,
    materialize_split,
    per_class_metrics_frame,
    prediction_metrics_frame,
    representation_stability_frame,
)

CELL = PenaltySweepCell(
    penalty_metric="spectral",
    input_geometry="fisher_rao",
    weight=1e-3,
    hinge_threshold=None,
    hinge_alpha=1.0,
    penalized_space="u",
    calculation_method="exact_autograd_jacobian",
)


class _StubBinner:
    """Binner exposing the batched transform and the config the cache key records."""

    def __init__(self, bins: int) -> None:
        self.bins = bins
        self._config = {"bins": bins}

    def transform(self, raw_batch):
        return _StubDenseBatch(raw_batch.intensities)


class _StubRawBatch:
    """What the reader hands back: raw intensities before binning."""

    def __init__(self, intensities: torch.Tensor) -> None:
        self.intensities = intensities


class _StubDenseBatch:
    """What the binner hands back: dense spectra on the binning grid."""

    def __init__(self, spectra: torch.Tensor) -> None:
        self.spectra = spectra


class _StubContext:
    def __init__(self, binner) -> None:
        self.binner = binner


class _StubTargetBatch:
    def __init__(self, values, masks) -> None:
        self.values = values
        self.masks = masks


class _StubDataset:
    """Dataset exposing the batched read path `materialize_split` uses."""

    def __init__(self, size: int, bins: int, classes: int, target_field: str) -> None:
        self.size = size
        self.bins = bins
        self.classes = classes
        self.target_field = target_field
        self.normalization = "none"
        self.active_context = _StubContext(_StubBinner(bins))
        self.accessed: list[int] = []
        self.batch_calls = 0

    def __len__(self) -> int:
        return self.size

    def _spectrum(self, index: int) -> torch.Tensor:
        return torch.full((self.bins,), float(index))

    def __getitem__(self, index: int):
        self.accessed.append(index)
        targets = {self.target_field: torch.zeros(self.classes, dtype=torch.float32)}
        targets[self.target_field][index % self.classes] = 1.0
        masks = {self.target_field: torch.ones(self.classes, dtype=torch.float32)}
        return index, self._spectrum(index), targets, masks

    def get_raw_batch(self, indices):
        self.batch_calls += 1
        self.accessed.extend(int(index) for index in indices)
        return _StubRawBatch(torch.stack([self._spectrum(int(i)) for i in indices]))

    def get_target_batch(self, indices):
        values = torch.zeros((len(indices), self.classes), dtype=torch.float32)
        for row, index in enumerate(indices):
            values[row, int(index) % self.classes] = 1.0
        masks = torch.ones((len(indices), self.classes), dtype=torch.float32)
        return _StubTargetBatch({self.target_field: values}, {self.target_field: masks})

    def normalize_batch(self, spectra: torch.Tensor) -> torch.Tensor:
        return spectra


class _StubPartitions:
    def __init__(self, **datasets) -> None:
        for name, dataset in datasets.items():
            setattr(self, name, dataset)


def _split(name: str, targets: np.ndarray, mask: np.ndarray) -> MaterializedSplit:
    samples = targets.shape[0]
    return MaterializedSplit(
        name=name,
        spectra=torch.zeros((samples, 4)),
        targets=targets,
        mask=mask,
        indices=np.arange(samples),
        total=samples,
        fraction=1.0,
        seed=42,
    )


class TestMaterializeSplit:
    """One-time decoding of a split."""

    def test_subsample_is_deterministic_and_records_its_indices(self) -> None:
        partitions = _StubPartitions(
            train=_StubDataset(100, bins=4, classes=3, target_field="molecule")
        )

        first = materialize_split(partitions, "train", "molecule", fraction=0.1, seed=7)
        second = materialize_split(partitions, "train", "molecule", fraction=0.1, seed=7)

        assert first.sampled == 10
        assert first.total == 100
        np.testing.assert_array_equal(first.indices, second.indices)
        # Indices are the reproducible record, independent of the random stream.
        assert list(first.indices) == sorted(first.indices)

    def test_each_pixel_is_decoded_exactly_once(self) -> None:
        dataset = _StubDataset(50, bins=4, classes=3, target_field="molecule")
        partitions = _StubPartitions(train=dataset)

        split = materialize_split(partitions, "train", "molecule", fraction=0.2, seed=1)

        assert len(dataset.accessed) == split.sampled
        assert len(set(dataset.accessed)) == split.sampled

    def test_shapes_follow_the_dataset(self) -> None:
        partitions = _StubPartitions(
            test=_StubDataset(20, bins=6, classes=5, target_field="molecule")
        )

        split = materialize_split(partitions, "test", "molecule", fraction=0.5, seed=3)

        assert split.spectra.shape == (10, 6)
        assert split.targets.shape == (10, 5)
        assert split.mask.shape == (10, 5)

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_invalid_fraction_is_rejected(self, fraction: float) -> None:
        partitions = _StubPartitions(
            train=_StubDataset(10, bins=4, classes=2, target_field="molecule")
        )

        with pytest.raises(ValueError):
            materialize_split(partitions, "train", "molecule", fraction=fraction)


class TestFairScopeMask:
    """Class selection shared by the compared models."""

    def test_only_classes_positive_in_both_splits_are_selected(self) -> None:
        train = _split(
            "train",
            targets=np.array([[1.0, 1.0, 0.0, 0.0]]),
            mask=np.ones((1, 4)),
        )
        test = _split(
            "test",
            targets=np.array([[1.0, 0.0, 1.0, 0.0]]),
            mask=np.ones((1, 4)),
        )

        mask = fair_scope_mask(train, test)

        np.testing.assert_array_equal(mask, [True, False, False, False])

    def test_masked_out_positives_do_not_count_as_active(self) -> None:
        # A positive whose label is unavailable cannot have been learned from.
        train = _split(
            "train",
            targets=np.array([[1.0, 1.0]]),
            mask=np.array([[1.0, 0.0]]),
        )
        test = _split("test", targets=np.array([[1.0, 1.0]]), mask=np.ones((1, 2)))

        mask = fair_scope_mask(train, test)

        np.testing.assert_array_equal(mask, [True, False])


class TestPredictionMetricsFrame:
    """Long-form prediction records."""

    def _records(self, scopes=None):
        targets = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        split = _split("test", targets=targets, mask=np.ones_like(targets))
        logits = np.array([[3.0, -3.0], [-3.0, 3.0], [3.0, 3.0], [-3.0, -3.0]])
        return prediction_metrics_frame(
            logits,
            split,
            CELL,
            model_name="campaign__task_000000",
            campaign="campaign",
            repetition=2,
            scopes=scopes,
        )

    def test_every_evaluate_head_metric_is_emitted(self) -> None:
        rows = self._records()

        emitted = {row["metric"] for row in rows}
        assert {"average_precision", "roc_auc", "macro_f1", "micro_f1"} <= emitted
        # Threshold-dependent and ranking metrics must both survive, since they
        # respond differently to a latent-rescaling penalty.
        assert {"macro_recall", "macro_precision", "hamming_loss"} <= emitted

    def test_grid_identity_travels_with_every_row(self) -> None:
        row = self._records()[0]

        assert row["campaign"] == "campaign"
        assert row["repetition"] == 2
        assert row["input_geometry"] == "fisher_rao"
        assert row["cell_label"] == CELL.label
        assert row["split"] == "test"

    def test_scopes_are_recorded_separately_with_their_class_counts(self) -> None:
        rows = self._records(
            scopes={
                ALL_CLASSES_SCOPE: None,
                FAIR_SCOPE: np.array([True, False]),
            }
        )

        counts = {row["scope"]: row["class_count"] for row in rows}
        assert counts[ALL_CLASSES_SCOPE] == 2
        assert counts[FAIR_SCOPE] == 1


class TestPerClassMetricsFrame:
    """Per-class prediction records."""

    def test_class_indices_refer_to_the_original_axis_under_a_mask(self) -> None:
        # Without remapping, a masked evaluation would report class 0 for what is
        # really class 2, making per-class rows unjoinable across scopes.
        targets = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        split = _split("test", targets=targets, mask=np.ones_like(targets))
        logits = np.array([[0.0, 0.0, 4.0], [0.0, 0.0, -4.0]])

        rows = per_class_metrics_frame(
            logits,
            split,
            CELL,
            model_name="campaign__task_000000",
            campaign="campaign",
            repetition=0,
            class_mask=np.array([False, False, True]),
        )

        assert len(rows) == 1
        assert rows[0]["class_index"] == 2
        assert rows[0]["positive_samples"] == 1


class TestLatentGeometryFrame:
    """Latent cloud summaries."""

    def _rows(self, latent: np.ndarray):
        return latent_geometry_frame(
            latent,
            CELL,
            split_name="test",
            model_name="campaign__task_000000",
            campaign="campaign",
            repetition=0,
        )

    def test_expected_statistics_are_emitted(self) -> None:
        generator = np.random.default_rng(0)
        latent = generator.normal(size=(200, 6))

        metrics = {row["metric"] for row in self._rows(latent)}

        assert metrics == {
            "cloud_asymmetry",
            "trace",
            "effective_rank",
            "participation_ratio",
            "two_nn_intrinsic_dimension",
        }

    def test_two_nn_estimate_above_the_ambient_dimension_is_discarded(self) -> None:
        # Regression: a collapsed latent produced a TwoNN estimate of 127.7 on a
        # 10-dimensional latent, which would otherwise be plotted as a real value.
        generator = np.random.default_rng(1)
        collapsed = generator.normal(size=(150, 4)) * 1e-9
        collapsed[:, 0] += np.linspace(0.0, 1e-8, 150)

        rows = {row["metric"]: row["value"] for row in self._rows(collapsed)}

        if rows["two_nn_intrinsic_dimension"] is not None:
            assert rows["two_nn_intrinsic_dimension"] <= collapsed.shape[1]

    def test_a_well_behaved_latent_keeps_its_two_nn_estimate(self) -> None:
        generator = np.random.default_rng(2)
        latent = generator.normal(size=(300, 8))

        rows = {row["metric"]: row["value"] for row in self._rows(latent)}

        assert rows["two_nn_intrinsic_dimension"] is not None
        assert 0 < rows["two_nn_intrinsic_dimension"] <= 8


class TestRepresentationStabilityFrame:
    """Cross-repetition representational agreement."""

    def test_every_unordered_repetition_pair_is_compared(self) -> None:
        generator = np.random.default_rng(3)
        latents = {index: generator.normal(size=(80, 5)) for index in range(3)}

        rows = representation_stability_frame(
            latents, CELL, split_name="test", campaign="campaign"
        )

        pairs = {(row["repetition_a"], row["repetition_b"]) for row in rows}
        assert pairs == {(0, 1), (0, 2), (1, 2)}
        assert {row["metric"] for row in rows} == {
            "procrustes_distance",
            "linear_cka",
            "knn_overlap",
        }

    def test_identical_representations_are_maximally_similar(self) -> None:
        generator = np.random.default_rng(4)
        latent = generator.normal(size=(120, 5))

        rows = representation_stability_frame(
            {0: latent, 1: latent.copy()}, CELL, split_name="test", campaign="campaign"
        )
        values = {row["metric"]: row["value"] for row in rows}

        assert values["linear_cka"] == pytest.approx(1.0, abs=1e-6)
        assert values["knn_overlap"] == pytest.approx(1.0, abs=1e-6)
        assert values["procrustes_distance"] == pytest.approx(0.0, abs=1e-6)


class TestScratchWorkspaceLink:
    """Resolution of compute-node paths baked into saved configs."""

    def test_link_is_created_and_is_idempotent(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        scratch = tmp_path / "scratch" / "campaign" / "workspace"

        first = link_campaign_scratch_workspace(scratch, workspace)
        second = link_campaign_scratch_workspace(scratch, workspace)

        assert first.is_symlink()
        assert first.resolve() == workspace.resolve()
        assert second == first

    def test_existing_real_directory_is_not_replaced(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        scratch = tmp_path / "scratch" / "campaign" / "workspace"
        scratch.mkdir(parents=True)

        with pytest.raises(FileExistsError):
            link_campaign_scratch_workspace(scratch, workspace)


class _StubModel:
    """Model stub returning a fixed head output and latent for any batch."""

    def __init__(self, classes: int = 3, latent_dim: int = 4) -> None:
        self.classes = classes
        self.latent_dim = latent_dim
        self._parameter = torch.nn.Parameter(torch.zeros(1))
        self.eval_calls = 0
        # `encoder_layer_norm_parameters` reads the bottleneck LayerNorm affine off a
        # CNNEncoder-shaped model; the stub reproduces just that path.
        self.encoder = torch.nn.Module()
        self.encoder.bottleneck_layer = torch.nn.Sequential(torch.nn.LayerNorm(latent_dim))

    def parameters(self):
        yield self._parameter

    def eval(self):
        self.eval_calls += 1
        return self

    def __call__(self, batch):
        rows = batch.shape[0]
        generator = torch.Generator().manual_seed(int(batch.sum().item()) % 1000 + rows)
        return {
            "head_molecule_primary": torch.randn(rows, self.classes, generator=generator),
            "latent_space": torch.randn(rows, self.latent_dim, generator=generator),
        }


class TestEvaluateCampaignModels:
    """Whole-grid evaluation loop."""

    def _inputs(self, samples: int = 60):
        generator = np.random.default_rng(0)
        targets = (generator.random((samples, 3)) > 0.6).astype(float)
        split = _split("test", targets=targets, mask=np.ones_like(targets))
        record = {
            "campaign": "campaign-a",
            "model_name": "campaign-a__task_000000",
            "repetition": 0,
            "penalty_metric": "spectral",
            "input_geometry": "fisher_rao",
            "weight": 1e-3,
            "hinge_threshold": None,
            "hinge_alpha": 1.0,
            "penalized_space": "u",
            "calculation_method": "exact_autograd_jacobian",
        }
        return split, record

    def test_geometry_can_be_skipped_entirely(self) -> None:
        from msi_autoencoder_wrapper.analysis.autoencoder.experiments.sweep_evaluation import (
            evaluate_campaign_models,
        )

        split, record = self._inputs()
        model = _StubModel()

        result = evaluate_campaign_models(
            [record], {"test": split}, lambda name: model,
            head_name="molecule_primary", collect_per_class=False, collect_geometry=False,
        )

        assert result["prediction"]
        assert result["geometry"] == []
        assert result["latent_by_cell"] == {}

    def test_geometry_sample_size_caps_the_rows_used(self, monkeypatch) -> None:
        # The quadratic geometry measures are intractable on a full split across tens
        # of models, so the cap must actually reach the statistics.
        from msi_autoencoder_wrapper.analysis.autoencoder.experiments import (
            sweep_evaluation as module,
        )

        split, record = self._inputs(samples=60)
        seen_rows = []

        def _spy(latent, cell, **kwargs):
            seen_rows.append(latent.shape[0])
            return []

        monkeypatch.setattr(module, "latent_geometry_frame", _spy)
        module.evaluate_campaign_models(
            [record], {"test": split}, lambda name: _StubModel(),
            head_name="molecule_primary", collect_per_class=False,
            geometry_sample_size=25,
        )

        assert seen_rows == [25]

    def test_sample_size_above_the_split_uses_every_row(self, monkeypatch) -> None:
        from msi_autoencoder_wrapper.analysis.autoencoder.experiments import (
            sweep_evaluation as module,
        )

        split, record = self._inputs(samples=30)
        seen_rows = []
        monkeypatch.setattr(
            module, "latent_geometry_frame",
            lambda latent, cell, **kwargs: seen_rows.append(latent.shape[0]) or [],
        )
        module.evaluate_campaign_models(
            [record], {"test": split}, lambda name: _StubModel(),
            head_name="molecule_primary", collect_per_class=False,
            geometry_sample_size=500,
        )

        assert seen_rows == [30]


class _StubSubset:
    """Minimal `torch.utils.data.Subset` stand-in over a stub dataset."""

    def __init__(self, dataset, indices) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


class TestBatchedDecode:
    """The decode path must read in batches and address the owning dataset."""

    def test_pixels_are_requested_in_batches_not_one_at_a_time(self) -> None:
        dataset = _StubDataset(100, bins=4, classes=3, target_field="molecule")
        partitions = _StubPartitions(train=dataset)

        materialize_split(
            partitions, "train", "molecule", fraction=1.0, seed=1, decode_batch_size=32
        )

        # 100 pixels in batches of 32 is four reader calls, not one hundred.
        assert dataset.batch_calls == 4

    def test_subset_positions_are_translated_to_owning_dataset_indices(self) -> None:
        # `create_partitions` returns Subset views, so a split-local position is not
        # the index the owning dataset's batch reader expects.
        dataset = _StubDataset(100, bins=4, classes=3, target_field="molecule")
        subset = _StubSubset(dataset, [10, 20, 30, 40])
        partitions = _StubPartitions(train=subset)

        split = materialize_split(partitions, "train", "molecule", fraction=1.0, seed=1)

        assert split.sampled == 4
        # The stub encodes its own index into every spectrum value.
        assert sorted(split.spectra[:, 0].tolist()) == [10.0, 20.0, 30.0, 40.0]

    def test_batched_decode_matches_per_item_decode(self) -> None:
        dataset = _StubDataset(60, bins=5, classes=4, target_field="molecule")
        partitions = _StubPartitions(test=dataset)

        split = materialize_split(partitions, "test", "molecule", fraction=1.0, seed=2)

        expected = torch.stack([dataset[int(i)][1] for i in split.indices])
        torch.testing.assert_close(split.spectra, expected)


class TestDecodeCache:
    """A cached split must be reused only when it is provably the same split."""

    def test_second_call_reuses_the_cache_without_reading(self, tmp_path: Path) -> None:
        dataset = _StubDataset(80, bins=4, classes=3, target_field="molecule")
        partitions = _StubPartitions(test=dataset)

        first = materialize_split(
            partitions, "test", "molecule", fraction=1.0, seed=3, cache_directory=tmp_path
        )
        calls_after_first = dataset.batch_calls
        second = materialize_split(
            partitions, "test", "molecule", fraction=1.0, seed=3, cache_directory=tmp_path
        )

        assert dataset.batch_calls == calls_after_first
        torch.testing.assert_close(second.spectra, first.spectra)
        np.testing.assert_array_equal(second.targets, first.targets)
        np.testing.assert_array_equal(second.indices, first.indices)

    def test_a_different_seed_does_not_hit_the_cache(self, tmp_path: Path) -> None:
        dataset = _StubDataset(80, bins=4, classes=3, target_field="molecule")
        partitions = _StubPartitions(test=dataset)

        materialize_split(
            partitions, "test", "molecule", fraction=0.5, seed=3, cache_directory=tmp_path
        )
        calls_after_first = dataset.batch_calls
        materialize_split(
            partitions, "test", "molecule", fraction=0.5, seed=4, cache_directory=tmp_path
        )

        assert dataset.batch_calls > calls_after_first

    def test_a_changed_binner_configuration_invalidates_the_cache(self, tmp_path: Path) -> None:
        # The decisive case: same split, same seed, different binning. Reusing the
        # cache here would silently return spectra binned under the wrong grid.
        dataset = _StubDataset(80, bins=4, classes=3, target_field="molecule")
        partitions = _StubPartitions(test=dataset)
        materialize_split(
            partitions, "test", "molecule", fraction=1.0, seed=3, cache_directory=tmp_path
        )
        calls_after_first = dataset.batch_calls

        dataset.active_context.binner._config = {"bins": 999}
        materialize_split(
            partitions, "test", "molecule", fraction=1.0, seed=3, cache_directory=tmp_path
        )

        assert dataset.batch_calls > calls_after_first
