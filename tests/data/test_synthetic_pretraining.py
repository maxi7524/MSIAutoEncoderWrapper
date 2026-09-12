"""Synthetic labels, composition scope, axis coverage, and reproducibility."""

import numpy as np
import pytest
import torch

from msi_dataset_manager.annotations.candidates import (
    CandidateCatalogReader,
    CandidateCatalogWriter,
    CandidateClass,
    CandidateCompound,
    make_candidate_ions,
)
from msi_autoencoder_wrapper.data import TargetSchema
from msi_autoencoder_wrapper.data.annotation_evidence import IonCatalogue
from msi_autoencoder_wrapper.data.pretraining import (
    SyntheticPeakSource,
    CandidateCatalogPeakSource,
    SyntheticSpectrumConfig,
    SyntheticSpectrumDataset,
    SyntheticSamplingManager,
)
from msi_autoencoder_wrapper.training.criterions.autoencoder.pretraining.element_count_loss import ElementCountLoss


@pytest.fixture
def synthetic_factory():
    catalogue = IonCatalogue(("C2H4|+H", "C3H6|+H"), ((1,), (8,)), 10)
    schemas = {
        "molecule": TargetSchema("molecule", "multi_label", catalogue.class_names),
        "element_counts": TargetSchema("element_counts", "regression", ("C", "H")),
    }
    def build(modes, **kwargs):
        options = {"samples": 50, "modes": modes, **kwargs}
        return SyntheticSpectrumDataset(catalogue, torch.linspace(100, 3000, 10), schemas,
                                        SyntheticSpectrumConfig(**options))
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


def test_sampling_plan_has_exact_counts_and_can_mask_supervision(synthetic_factory):
    dataset = synthetic_factory(
        (),
        samples=5,
        sampling_plan=(
            {"strategy": "single_annotated", "count": 2},
            {"strategy": "single_random", "count": 3, "label_targets": False},
        ),
    )
    samples = [dataset[index] for index in range(len(dataset))]
    assert all(bool(sample[3]["molecule"].all()) for sample in samples[:2])
    assert all(sample[2]["molecule"].sum() == 1 for sample in samples[:2])
    assert all(not bool(sample[3]["molecule"].any()) for sample in samples[2:])


class CandidateLikePeakSource(SyntheticPeakSource):
    """Stand-in for a future candidate catalogue mapped onto the model axis."""

    feature_count = 10
    class_names = ("C2H4|+H", "C3H6|+H")
    bins = ((1,), (8,))


def test_dataset_accepts_a_non_annotation_peak_source():
    schemas = {
        "molecule": TargetSchema("molecule", "multi_label", CandidateLikePeakSource.class_names),
    }
    dataset = SyntheticSpectrumDataset(
        None,
        torch.linspace(100, 3000, 10),
        schemas,
        SyntheticSpectrumConfig(samples=1, sampling_plan=({"strategy": "single_annotated", "count": 1},)),
        peak_source=CandidateLikePeakSource(),
    )
    _, spectrum, values, masks = dataset[0]
    assert spectrum.shape == (10,)
    assert values["molecule"].sum() == 1
    assert bool(masks["molecule"].all())


def test_sampling_strategies_are_discoverable_with_constructor_configuration():
    available = SyntheticSamplingManager.get_available_strategies()
    assert "annotated" in available
    assert available["annotated"]["parameters"] == {"min_peaks": 1, "max_peaks": None}
    assert "labelled mixture" in available["annotated"]["docstring"]


class CandidateBinner:
    """Map selected candidate masses onto a fixed ten-bin synthetic axis."""

    def GetXAxis(self):
        return np.arange(10, dtype=np.float64)

    def map_mass_values_to_bins(self, masses):
        values = np.asarray(masses, dtype=np.float64)
        return np.where(values < 100, 1, np.where(values < 500, 8, -1))


def _candidate_source(tmp_path):
    """Create candidates with one provider duplicate and one distinct m/z."""
    compounds = (
        CandidateCompound(
            provider="HMDB",
            provider_version="5.0",
            identifier="hmdb-ethylene",
            name="Ethylene",
            formula="C2H4",
            classes=(CandidateClass("HMDB", "HMDB:alkene", "Alkene", 0),),
        ),
        CandidateCompound(
            provider="ChEBI",
            provider_version="255",
            identifier="CHEBI:18153",
            name="Ethylene",
            formula="C2H4",
            classes=(CandidateClass("ChEBI", "CHEBI:alkene", "Alkene", 0),),
        ),
        CandidateCompound(
            provider="LIPID_MAPS",
            provider_version="fixture",
            identifier="LMFA0001",
            name="Fixture glucose",
            formula="C6H12O6",
            classes=(CandidateClass("LIPID_MAPS", "LIPID_MAPS:sugar", "Sugar", 0),),
        ),
        CandidateCompound(
            provider="fixture",
            provider_version="1",
            identifier="outside-axis",
            name="Outside axis",
            formula="C100H2",
        ),
    )
    CandidateCatalogWriter().write(
        path=tmp_path / "candidates.sqlite",
        compounds=compounds,
        ions=(
            ion
            for compound in compounds
            for ion in make_candidate_ions(compound, polarity="Positive", adducts=("+H",))
        ),
        manifest={"schema_version": 1},
    )
    return CandidateCatalogPeakSource(
        CandidateCatalogReader(tmp_path / "candidates.sqlite"),
        CandidateBinner(),
    )


def test_candidate_peak_source_preserves_distinct_mz_and_aggregates_duplicates(tmp_path):
    """Provider duplicates become one provenance-rich label without mass merging."""
    source = _candidate_source(tmp_path)

    assert source.class_names == ("C2H4|+H", "C6H12O6|+H")
    assert source.bins == ((1,), (8,))
    metadata = source.get_label_metadata(0)
    assert metadata["providers"] == ("ChEBI", "HMDB")
    assert len(metadata["candidate_ion_keys"]) == 2
    assert metadata["chemical_classes"] == ("Alkene",)


def test_candidate_generator_expands_permutations_and_preserves_batch_metadata(tmp_path):
    """Candidate weights are normalized after independent occupancy/intensity draws."""
    source = _candidate_source(tmp_path)
    schemas = {
        "molecule": TargetSchema(
            "molecule",
            "multi_label",
            tuple(reversed(source.class_names)),
        ),
    }
    dataset = SyntheticSpectrumDataset(
        None,
        torch.arange(10, dtype=torch.float64),
        schemas,
        SyntheticSpectrumConfig(
            samples=6,
            sampling_plan=(
                {
                    "strategy": "candidate_permuted",
                    "count": 2,
                    "parameters": {
                        "permutations": 3,
                        "min_molecules": 2,
                        "max_molecules": 2,
                        "occupancy_alpha": 0.5,
                        "intensity_log_sigma": 0.4,
                    },
                },
            ),
        ),
        peak_source=source,
    )

    sample = dataset[0]
    assert sample[2]["molecule"].sum() == 2
    assert sample.metadata["generator"] == "candidate_permuted"
    assert sum(component["weight"] for component in sample.metadata["components"]) == pytest.approx(1.0)
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert len(batch.metadata["synthetic_generation"]) == 2
    assert all(
        item["mixing_mode"] == "weighted_sum"
        for item in batch.metadata["synthetic_generation"]
    )


def test_candidate_convolved_strategy_records_weighted_sum_configuration(tmp_path):
    """Class-aware candidate mixtures retain their generation configuration."""
    source = _candidate_source(tmp_path)
    schemas = {"molecule": TargetSchema("molecule", "multi_label", source.class_names)}
    dataset = SyntheticSpectrumDataset(
        None,
        torch.arange(10, dtype=torch.float64),
        schemas,
        SyntheticSpectrumConfig(
            samples=1,
            sampling_plan=(
                {
                    "strategy": "candidate_convolved",
                    "count": 1,
                    "parameters": {"min_molecules": 2, "max_molecules": 2},
                },
            ),
        ),
        peak_source=source,
    )

    sample = dataset[0]
    assert sample[2]["molecule"].sum() == 2
    assert sample.metadata["generator"] == "candidate_convolved"
    assert sample.metadata["mixing_mode"] == "convolution_weighted_sum"
