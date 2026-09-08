"""Synthetic labels, composition scope, axis coverage, and reproducibility."""

import numpy as np
import pytest
import torch

from msi_autoencoder_wrapper.data import TargetSchema
from msi_autoencoder_wrapper.data.annotation_evidence import IonCatalogue
from msi_autoencoder_wrapper.data.pretraining import SyntheticSpectrumConfig, SyntheticSpectrumDataset
from msi_autoencoder_wrapper.training.criterions.autoencoder.pretraining.element_count_loss import ElementCountLoss


@pytest.fixture
def synthetic_factory():
    catalogue = IonCatalogue(("C2H4|+H", "C3H6|+H"), ((1,), (8,)), 10)
    schemas = {
        "molecule": TargetSchema("molecule", "multi_label", catalogue.class_names),
        "element_counts": TargetSchema("element_counts", "regression", ("C", "H")),
    }
    def build(modes, **kwargs):
        return SyntheticSpectrumDataset(catalogue, torch.linspace(100, 3000, 10), schemas,
                                        SyntheticSpectrumConfig(samples=50, modes=modes, **kwargs))
    return build


def test_single_ion_targets_are_complete_and_counts_are_per_component(synthetic_factory):
    dataset = synthetic_factory(("single_annotated",))
    for sample in [dataset[i] for i in range(10)]:
        _, spectrum, values, masks = sample
        assert spectrum.shape == (10,) and spectrum.dtype == torch.float32
        torch.testing.assert_close(spectrum.sum(), torch.tensor(1.), rtol=1e-6, atol=1e-7)
        assert values["molecule"].sum() == 1 and bool(masks["molecule"].all())
        ion = int(values["molecule"].argmax())
        assert values["element_counts"].tolist() == ([2, 4] if ion == 0 else [3, 6])
        assert bool(masks["element_counts"].all())
        assert spectrum[dataset.catalogue.bins[ion][0]] > 0


def test_random_background_has_known_negative_ion_labels(synthetic_factory):
    dataset = synthetic_factory(("random",), max_peaks=5)
    for i in range(10):
        _, spectrum, values, masks = dataset[i]
        assert spectrum[1] == spectrum[8] == 0
        assert not bool(values["molecule"].any())
        assert bool(masks["molecule"].all())
        assert not bool(masks["element_counts"].any())
        assert int((spectrum > 0).sum()) <= 5


def test_mixtures_do_not_sum_element_counts(synthetic_factory):
    dataset = synthetic_factory(("annotated",), max_peaks=2)
    mixtures = [dataset[i] for i in range(len(dataset)) if dataset[i][2]["molecule"].sum() == 2]
    assert mixtures
    assert all(not bool(s[3]["element_counts"].any()) for s in mixtures)


def test_generator_is_epoch_reproducible_without_changing_global_rng(synthetic_factory):
    first = synthetic_factory(("mixed",), max_peaks=5)
    second = synthetic_factory(("mixed",), max_peaks=5)
    np.random.seed(19)
    reference = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    torch.testing.assert_close(first[4][1], second[4][1], rtol=0, atol=0)
    first.set_epoch(2)
    second.set_epoch(2)
    torch.testing.assert_close(first[4][1], second[4][1], rtol=0, atol=0)
    assert np.array_equal(reference[1], np.random.get_state()[1])
    assert torch.equal(torch_state, torch.get_rng_state())


def test_wide_axis_is_covered_without_extra_input_channels(synthetic_factory):
    dataset = synthetic_factory(("single_random",))
    batch = dataset.collate_fn([dataset[i] for i in range(len(dataset))])
    assert batch.model_input().shape == (50, 10)
    assert batch.spectra[:, 0].sum() > 0 and batch.spectra[:, -1].sum() > 0
    assert not batch.metadata


def test_composition_loss_ignores_real_or_mixed_targets(synthetic_factory):
    dataset = synthetic_factory(("random",))
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    logits = torch.zeros(2, 2, requires_grad=True)
    loss = ElementCountLoss("composition", "element_counts")({"head_composition": logits}, batch)
    loss.backward()
    assert loss == 0 and not bool(logits.grad.any())
