"""Chemical class aggregation preserves candidate ambiguity and PU semantics."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from msi_dataset_manager.annotations.chemistry import SnapshotProvider, enrich_annotations
from msi_dataset_manager.annotations.index import build_annotation_index
from msi_autoencoder_wrapper.binners.binners_strategies.linear_binner import LinearBinning
from msi_autoencoder_wrapper.models.datasets.strategies.pixel_dataset import PixelDataset


@pytest.fixture
def chemical_dataset(tmp_path):
    class Reader:
        def GetNumberOfSpectra(self):
            return 3
        def GetSpectrum(self, index):
            return np.array([200.1, 200.8]), np.array([1., float(index % 2)])
    class Annotations:
        def get_spectrum_annotation_index(self, _=None):
            return build_annotation_index(spectrum_ids=[0, 1, 2], entries={
                0: [(("C2H4", "+H"), 200.1)],
                1: [(("C2H4", "+H"), 200.1), (("C3H6", "+H"), 200.8)],
                2: [],
            })
        def get_dataset_metadata(self):
            return {}
        def get_annotations(self):
            return []
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"provider": "fixture", "version": "1", "candidates": [
        {"identifier": "a", "formula": "C2H4", "classes": ["lipid", "PC"]},
        {"identifier": "b", "formula": "C2H4", "classes": ["lipid", "PE"]},
        {"identifier": "c", "formula": "C3H6", "classes": ["lipid", "PC"]},
    ]}))
    db = tmp_path / "chemistry.sqlite"
    enrich_annotations([{"formula": "C2H4", "adduct": "+H"}, {"formula": "C3H6", "adduct": "+H"}], db, SnapshotProvider(snapshot))
    context = SimpleNamespace(annotation_reader=Annotations(),
                              binner=LinearBinning(bin_step=.5, x_min=200, x_max=201),
                              get_data_reader=lambda _: Reader())
    return PixelDataset(active_context=context, normalization="tic",
                        target_specs={"molecule": {"type": "multi_label"},
                                      "chemical_class": {"type": "multi_label"},
                                      "element_counts": {"type": "regression"}},
                        chemistry={"path": str(db), "provider": "fixture", "version": "1"})


def test_real_classes_use_common_ancestors_without_counting_ions(chemical_dataset):
    dataset = chemical_dataset
    assert dataset.get_target_schemas()["chemical_class"].class_names == ("PC", "PE", "lipid")
    first = dataset[0]
    assert first[2]["chemical_class"].tolist() == [0., 0., 1.]
    assert first[3]["chemical_class"].tolist() == [False, False, True]
    second = dataset[1]
    assert second[2]["chemical_class"].tolist() == [1., 0., 1.]
    assert second[3]["chemical_class"].tolist() == [True, False, True]
    assert not bool(second[3]["element_counts"].any())
    empty = dataset[2]
    assert not bool(empty[2]["chemical_class"].any())
    assert not bool(empty[3]["chemical_class"].any())


def test_explicit_source_selection_is_stable_and_validated(chemical_dataset):
    dataset = chemical_dataset
    dataset.subset({"method": "source_indices", "indices": [2, 0]})
    assert [dataset.get_sample_id(i) for i in range(len(dataset))] == [0, 2]
    with pytest.raises(ValueError, match="unique"):
        dataset.subset({"method": "source_indices", "indices": [0, 0]})


def test_real_spectra_are_zero_padded_on_a_wider_axis(chemical_dataset):
    dataset = chemical_dataset
    dataset.active_context.binner = LinearBinning(bin_step=.5, x_min=100, x_max=3000)
    dataset.configure_annotations({})
    spectrum = dataset[0][1]
    assert spectrum.ndim == 1
    assert not bool(spectrum[:200].any())
    assert not bool(spectrum[202:].any())
    torch.testing.assert_close(spectrum.sum(), torch.tensor(1.), rtol=1e-6, atol=1e-7)


def test_synthetic_chemical_targets_mask_ambiguous_classes(chemical_dataset):
    from msi_autoencoder_wrapper.data.annotation_evidence import IonCatalogue
    from msi_autoencoder_wrapper.data.pretraining.synthetic import SyntheticSpectrumConfig, SyntheticSpectrumDataset

    dataset = chemical_dataset
    common = dict(catalogue=IonCatalogue.from_dataset(dataset),
                  mass_axis=torch.as_tensor(dataset.active_context.binner.GetXAxis()),
                  schemas=dataset.get_target_schemas(), chemistry=dataset.get_chemical_descriptions(),
                  config=SyntheticSpectrumConfig(samples=1, modes=("single_annotated",)))
    ambiguous = SyntheticSpectrumDataset(eligible_ions=(0,), **common)[0]
    assert ambiguous[2]["chemical_class"].tolist() == [0., 0., 1.]
    assert ambiguous[3]["chemical_class"].tolist() == [False, False, True]
    certain = SyntheticSpectrumDataset(eligible_ions=(1,), **common)[0]
    assert certain[2]["chemical_class"].tolist() == [1., 0., 1.]
    assert certain[3]["chemical_class"].tolist() == [True, True, True]
    assert certain[3]["element_counts"].all()
