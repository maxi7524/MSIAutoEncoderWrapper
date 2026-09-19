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
    AnnotationPeakRecord,
    AnnotationPopulation,
    SyntheticPeakSource,
    CandidateCatalogPeakSource,
    SyntheticSpectrumConfig,
    SyntheticSpectrumDataset,
    SyntheticSamplingManager,
)
from msi_autoencoder_wrapper.data.pretraining.representations import (
    SyntheticRepresentationContext,
    get_representation_strategy,
)
from msi_autoencoder_wrapper.data.pretraining.sampling import (
    SyntheticComponent,
    SyntheticSampleDefinition,
)
from msi_autoencoder_wrapper.data.supervision_masks import simulated_negative_mask_key
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
        batch = dataset.collate_fn([dataset[i]])
        assert bool(batch.targets.masks[simulated_negative_mask_key("molecule")].all())


def test_synthetic_labels_emit_positive_and_simulated_negative_masks(synthetic_factory):
    """Generator-owned labels distinguish present ions from known absences."""
    dataset = synthetic_factory(("single_annotated",))
    batch = dataset.collate_fn([dataset[0]])
    values = batch.targets.values["molecule"][0]
    simulated_negative = batch.targets.masks[simulated_negative_mask_key("molecule")][0]
    assert int((values > 0.5).sum()) == 1
    assert int(simulated_negative.sum()) == 1
    assert not bool(simulated_negative[values > 0.5].any())


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


def _annotation_population():
    return AnnotationPopulation(
        feature_count=10,
        spectrum_ids=(17, 18),
        records=(
            AnnotationPeakRecord(17, 1, (0, 1)),
            AnnotationPeakRecord(18, 8, (1,)),
        ),
    )


def test_pixel_aware_annotation_mixture_preserves_complete_local_labels():
    catalogue = IonCatalogue(("C2H4|+H", "C3H6|+H"), ((1,), (8,)), 10)
    dataset = SyntheticSpectrumDataset(
        catalogue,
        torch.linspace(100, 3000, 10),
        {"molecule": TargetSchema("molecule", "multi_label", catalogue.class_names)},
        SyntheticSpectrumConfig(
            samples=1,
            sampling_plan=(
                {
                    "strategy": "annotation_uniform_mixture",
                    "count": 1,
                    "parameters": {"min_peaks": 1, "max_peaks": 1},
                },
            ),
        ),
        annotation_population=_annotation_population(),
    )

    sample = dataset[0]
    assert sample[2]["molecule"].tolist() == [1.0, 1.0]
    assert bool(sample[3]["molecule"].all())
    assert sample.metadata["generator"] == "annotation_uniform_mixture"


def test_annotation_mixture_schedules_requested_classes_uniformly_per_epoch():
    catalogue = IonCatalogue(("C2H4|+H", "C3H6|+H"), ((1,), (8,)), 10)
    dataset = SyntheticSpectrumDataset(
        catalogue,
        torch.linspace(100, 3000, 10),
        {"molecule": TargetSchema("molecule", "multi_label", catalogue.class_names)},
        SyntheticSpectrumConfig(
            samples=4,
            sampling_plan=(
                {
                    "strategy": "annotation_uniform_mixture",
                    "count": 4,
                    "parameters": {"min_peaks": 1, "max_peaks": 1},
                },
            ),
        ),
        annotation_population=_annotation_population(),
    )

    requested = [dataset[index].metadata["requested_labels"][0] for index in range(4)]
    assert requested.count(0) == requested.count(1) == 2
    dataset.set_epoch(3)
    epoch_three = [dataset[index].metadata["requested_labels"] for index in range(4)]
    dataset.set_epoch(4)
    dataset.set_epoch(3)
    assert [dataset[index].metadata["requested_labels"] for index in range(4)] == epoch_three


def test_axis_coverage_requires_an_integer_number_of_full_axis_passes():
    catalogue = IonCatalogue(("C2H4|+H", "C3H6|+H"), ((1,), (8,)), 10)
    dataset = SyntheticSpectrumDataset(
        catalogue,
        torch.linspace(100, 3000, 10),
        {"molecule": TargetSchema("molecule", "multi_label", catalogue.class_names)},
        SyntheticSpectrumConfig(
            samples=3,
            sampling_plan=({"strategy": "annotation_axis_coverage", "count": 3},),
        ),
        annotation_population=_annotation_population(),
    )

    with pytest.raises(ValueError, match="divisible"):
        dataset[0]


def test_plan_entry_can_override_the_phase_representation(synthetic_factory):
    dataset = synthetic_factory(
        (),
        samples=1,
        peak_radius=0,
        sampling_plan=(
            {
                "strategy": "single_annotated",
                "count": 1,
                "representation": {
                    "strategy": "triangular_peak",
                    "parameters": {"peak_radius": 2},
                },
            },
        ),
    )
    spectrum = dataset[0][1]
    assert int((spectrum > 0).sum()) == 4


class IsotopeEnvelopePeakSource(SyntheticPeakSource):
    """Two molecular ions sufficiently separated on a dense m/z axis."""

    feature_count = 2_501
    class_names = ("C2H4|+H", "C6H12O6|+H")
    bins = ((290,), (1810,))


def _isospec_mass_to_bin(masses):
    """Map a 0.1 m/z test axis with out-of-range values rejected."""
    values = np.asarray(masses, dtype=np.float64)
    indices = np.rint(values * 10).astype(np.int64)
    return np.where((indices >= 0) & (indices <= 2_500), indices, -1)


def test_isospec_representation_renders_adduct_envelopes_and_sums_mixtures():
    """Fine isotope lines are binned through the active binner before mixing."""
    source = IsotopeEnvelopePeakSource()
    context = SyntheticRepresentationContext(
        source=source,
        feature_count=source.feature_count,
        mass_axis=np.arange(source.feature_count, dtype=np.float64) / 10,
        mass_to_bin=_isospec_mass_to_bin,
    )
    definition = SyntheticSampleDefinition(
        components=(
            SyntheticComponent(center=290, label_indices=(0,), intensity_weight=1.0),
            SyntheticComponent(center=1811, label_indices=(1,), intensity_weight=1.0),
        ),
        label_targets=True,
        metadata={},
    )

    renderer = get_representation_strategy(
        "isospec_envelope",
        probability_coverage=0.999,
    )
    spectrum = renderer.render(np.random.default_rng(42), context, definition)

    assert spectrum.shape == (source.feature_count,)
    assert np.isfinite(spectrum).all()
    assert np.all(spectrum >= 0)
    assert spectrum.sum() > 0
    assert spectrum[290] > 0
    assert spectrum[1811] > 0


def test_isospec_representation_requires_declared_molecular_components():
    """A non-molecular reconstruction sample cannot invent an isotope envelope."""
    source = IsotopeEnvelopePeakSource()
    context = SyntheticRepresentationContext(
        source=source,
        feature_count=source.feature_count,
        mass_axis=np.arange(source.feature_count, dtype=np.float64) / 10,
    )
    definition = SyntheticSampleDefinition(
        components=(SyntheticComponent(center=100, intensity_weight=1.0),),
        label_targets=False,
        metadata={},
    )

    with pytest.raises(ValueError, match="declared molecular components"):
        get_representation_strategy("isospec_envelope").render(
            np.random.default_rng(42), context, definition
        )


def test_dataset_selects_isospec_representation_from_its_sampling_plan():
    """The phase configuration reaches the registered molecular renderer."""
    source = IsotopeEnvelopePeakSource()
    dataset = SyntheticSpectrumDataset(
        None,
        torch.arange(source.feature_count, dtype=torch.float64) / 10,
        {"molecule": TargetSchema("molecule", "multi_label", source.class_names)},
        SyntheticSpectrumConfig(
            samples=1,
            representation={"strategy": "isospec_envelope"},
            sampling_plan=({"strategy": "single_annotated", "count": 1},),
        ),
        peak_source=source,
        mass_to_bin=_isospec_mass_to_bin,
    )

    _, spectrum, values, masks = dataset[0]
    assert spectrum.shape == (source.feature_count,)
    assert torch.isfinite(spectrum).all()
    assert spectrum.sum() == pytest.approx(1.0)
    assert values["molecule"].sum() == 1
    assert bool(masks["molecule"].all())


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
