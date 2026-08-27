"""Tests for sparse annotation mapping owned by model datasets."""

from __future__ import annotations

import numpy as np
import torch

from msi_dataset_manager.annotations.index import build_annotation_index

from msi_autoencoder_wrapper.binners.binners_strategies.linear_binner import LinearBinning
from msi_autoencoder_wrapper.models.datasets.annotations.config import (
    AnnotationTargetSettings,
)
from msi_autoencoder_wrapper.models.datasets.annotations.manager import DatasetAnnotationManager
from msi_autoencoder_wrapper.models.datasets.strategies.pixel_dataset import PixelDataset


class _AnnotationReader:
    """Expose a deterministic reader-owned raw annotation CSR index."""

    def get_spectrum_annotation_index(self, _dataset_id):
        """Return annotations with both retained and out-of-range m/z values."""
        return build_annotation_index(
            spectrum_ids=[0, 1, 2],
            entries={
                0: [(("A", "+H"), 200.10), (("B", "+H"), 205.0)],
                1: [(("A", "+H"), 200.90)],
                2: [(("B", "+H"), 199.90)],
            },
        )


class _Context:
    """Minimal dataset context required by the annotation manager."""

    annotation_reader = _AnnotationReader()
    binner = LinearBinning(bin_step=0.5, x_min=200.0, x_max=201.0)


class _Dataset:
    """Minimal dataset owner used to isolate annotation mapping behaviour."""

    active_context = _Context()


def test_annotation_target_settings_use_predictive_default() -> None:
    """An omitted empty-spectrum policy uses the declared predictive default."""
    assert AnnotationTargetSettings.from_config({}) == AnnotationTargetSettings()


def test_annotation_mapping_filters_outside_binner_range_and_excludes_empty_rows() -> None:
    """Only annotations surviving the exact binner mapping define targets and rows."""
    manager = DatasetAnnotationManager(
        _Dataset(),
        {
            "mapping": {"x_mapping": "binner"},
            "targets": {"molecule": {"empty_spectrum_policy": "exclude"}},
        },
    )

    index = manager.get_mapped_index()

    assert index.spectrum_ids.tolist() == [0, 1]
    assert index.coordinates_for_spectrum(0).tolist() == [0]
    assert index.coordinates_for_spectrum(1).tolist() == [1]
    assert not index.has_annotations(2)
    assert manager.get_class_names() == ("A|+H",)
    assert manager.resolve_classes(["A|+H"]) == (0,)
    assert manager.get_selected_source_indices(3).tolist() == [0, 1]


def test_linear_binner_annotation_mapping_matches_forward_intervals() -> None:
    """The exact endpoint convention is shared by spectra and annotations."""
    binner = LinearBinning(bin_step=0.5, x_min=200.0, x_max=201.0)

    mapped = binner.map_mass_values_to_bins(
        np.array([199.9, 200.0, 200.49, 200.5, 201.0, 201.1, np.nan])
    )

    assert mapped.tolist() == [-1, 0, 0, 1, 1, -1, -1]


def test_linear_binner_annotation_mapping_matches_non_divisible_axis() -> None:
    """Annotation mapping retains the forward convention on a partial final interval."""
    binner = LinearBinning(bin_step=0.55, x_min=200.0, x_max=900.0)

    mapped = binner.map_mass_values_to_bins(np.array([899.99, 900.0]))
    expected = np.floor((np.array([899.99, 900.0]) - 200.0) / 0.55).astype(int)

    assert mapped.tolist() == expected.tolist()


def test_pixel_dataset_excludes_spectra_without_retained_annotations() -> None:
    """Annotation exclusion changes the public dataset index before train splitting."""

    class Reader:
        def GetNumberOfSpectra(self):
            return 3

        def GetSpectrum(self, spectrum_id):
            return np.array([200.1]), np.array([float(spectrum_id + 1)])

    class Context(_Context):
        @staticmethod
        def get_data_reader(_source):
            return Reader()

    dataset = PixelDataset(
        active_context=Context(),
        normalization="none",
        target_specs={"molecule": {"type": "multi_label"}},
        annotation_settings={
            "mapping": {"x_mapping": "binner"},
            "targets": {"molecule": {"empty_spectrum_policy": "exclude"}},
        },
    )

    assert len(dataset) == 2
    assert [dataset.get_sample_id(index) for index in range(len(dataset))] == [0, 1]


def test_pixel_dataset_masks_only_deterministic_train_positive_entries() -> None:
    """Configured positive masking leaves validation and test labels unchanged."""

    class Reader:
        def GetNumberOfSpectra(self):
            return 4

        def GetSpectrum(self, spectrum_id):
            return np.array([200.1]), np.array([float(spectrum_id + 1)])

    class AnnotationReader:
        @staticmethod
        def get_dataset_metadata():
            return {}

        @staticmethod
        def get_annotations():
            return [{"formula": "A", "adduct": "+H"}]

        def get_spectrum_annotation_index(self, _dataset_id):
            return build_annotation_index(
                spectrum_ids=[0, 1, 2, 3],
                entries={index: [(('A', '+H'), 200.1)] for index in range(4)},
            )

    class Context:
        annotation_reader = AnnotationReader()
        binner = LinearBinning(bin_step=0.5, x_min=200.0, x_max=201.0)

        @staticmethod
        def get_data_reader(_source):
            return Reader()

    settings = {
        "mapping": {"x_mapping": "binner"},
        "targets": {
            "molecule": {
                "empty_spectrum_policy": "exclude",
                "train_positive_mask": {"fraction": 0.5, "seed": 123},
            }
        },
    }
    split = {
        "strategy": "predefined",
        "fractions": {"train": 0.75, "validation": 0.0, "test": 0.25},
        "assignments": {"train": [0, 1, 2], "validation": [], "test": [3]},
    }
    dataset = PixelDataset(
        active_context=Context(),
        normalization="none",
        target_specs={"molecule": {"type": "multi_label"}},
        annotation_settings=settings,
        split=split,
    )

    partitions = dataset.create_partitions()
    train_targets = dataset.get_target_batch(partitions.train.indices).values["molecule"]
    test_targets = dataset.get_target_batch(partitions.test.indices).values["molecule"]

    assert int(train_targets.sum()) == 1
    assert torch.equal(test_targets, torch.tensor([[1.0]]))

    repeated = PixelDataset(
        active_context=Context(),
        normalization="none",
        target_specs={"molecule": {"type": "multi_label"}},
        annotation_settings=settings,
        split=split,
    )
    repeated_partitions = repeated.create_partitions()
    repeated_targets = repeated.get_target_batch(repeated_partitions.train.indices)
    assert torch.equal(train_targets, repeated_targets.values["molecule"])
