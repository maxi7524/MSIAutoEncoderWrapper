"""Tests for global candidate dictionaries built from dataset catalogues."""

from __future__ import annotations

import numpy as np
import torch

from msi_dataset_manager.annotations.candidates import (
    CandidateCatalogReader,
    CandidateCatalogWriter,
    CandidateCompound,
    make_candidate_ions,
)
from msi_autoencoder_wrapper.models.architectures.types.deconvolution import (
    GlobalCandidateDictionary,
)


class FixtureBinner:
    """Map fixture candidate masses onto a compact regular model axis."""

    def GetXAxis(self):
        """Return the ten-bin fixture axis."""
        return np.arange(10, dtype=np.float64)

    def map_mass_values_to_bins(self, masses):
        """Map the first fixture mass to bin one and the second to bin eight."""
        values = np.asarray(masses, dtype=np.float64)
        return np.where(values < 100.0, 1, np.where(values < 500.0, 8, -1))


def _catalog(tmp_path) -> CandidateCatalogReader:
    """Create a compact dataset-like candidate-ion SQLite fixture."""
    compounds = (
        CandidateCompound("fixture", "1", "one", "One", "C2H4"),
        CandidateCompound("fixture", "1", "two", "Two", "C6H12O6"),
        CandidateCompound("fixture", "1", "outside", "Outside", "C100H2"),
    )
    path = tmp_path / "candidates.sqlite"
    CandidateCatalogWriter().write(
        path=path,
        compounds=compounds,
        ions=tuple(
            ion
            for compound in compounds
            for ion in make_candidate_ions(compound, polarity="Positive", adducts=("+H",))
        ),
        manifest={"schema_version": 1},
    )
    return CandidateCatalogReader(path)


def test_dictionary_maps_all_catalogue_ions_inside_the_active_axis(tmp_path) -> None:
    """One dictionary column is created for every globally available ion."""
    dictionary = GlobalCandidateDictionary.from_candidate_catalog(_catalog(tmp_path), FixtureBinner())

    assert dictionary.matrix.shape == (10, 2)
    assert dictionary.mass_axis.shape == (10,)
    assert torch.equal(dictionary.matrix[:, 0], torch.nn.functional.one_hot(torch.tensor(1), 10).float())
    assert torch.equal(dictionary.matrix[:, 1], torch.nn.functional.one_hot(torch.tensor(8), 10).float())
    assert [record["formula"] for record in dictionary.candidate_ions] == ["C2H4", "C6H12O6"]


def test_dictionary_can_move_without_changing_candidate_provenance(tmp_path) -> None:
    """The global numerical matrix is movable while catalogue IDs remain stable."""
    dictionary = GlobalCandidateDictionary.from_candidate_catalog(_catalog(tmp_path), FixtureBinner())
    moved = dictionary.to("cpu")

    assert moved.matrix.device.type == "cpu"
    assert moved.candidate_ions == dictionary.candidate_ions


def test_dictionary_can_select_a_seeded_subset_before_dense_materialization(tmp_path) -> None:
    """Small feasibility runs do not allocate every catalogue ion on the axis."""
    catalog = _catalog(tmp_path)

    first = GlobalCandidateDictionary.from_candidate_catalog(
        catalog,
        FixtureBinner(),
        candidate_limit=1,
        selection_seed=17,
    )
    second = GlobalCandidateDictionary.from_candidate_catalog(
        catalog,
        FixtureBinner(),
        candidate_limit=1,
        selection_seed=17,
    )

    assert first.matrix.shape == (10, 1)
    assert first.candidate_ions == second.candidate_ions
    torch.testing.assert_close(first.matrix, second.matrix)
