"""Tests for reusable deconvolution notebook experiment computations."""

from __future__ import annotations

import torch

from msi_autoencoder_wrapper.models.architectures.types.deconvolution import (
    GlobalCandidateDictionary,
    SyntheticDeconvolutionConfig,
    catalogue_condition_counts,
    identifiability_experiment,
    projected_gradient_convergence,
    projected_gradient_gradcheck,
    sample_global_subdictionary,
)
from msi_autoencoder_wrapper.models.architectures.types.deconvolution.evaluation.catalogue_precompute import (
    catalogue_path,
    run_command,
)


class FixtureCatalog:
    """Return small candidate populations filtered by one METASPACE condition."""

    def get_candidate_ions(self, filters):
        """Return fixture ions based on requested polarity."""
        if filters["polarity"] == "Positive":
            return [
                {"compound_key": "one", "formula": "C2H4", "adduct": "+H"},
                {"compound_key": "two", "formula": "C6H12O6", "adduct": "+Na"},
            ]
        return [{"compound_key": "three", "formula": "C3H6", "adduct": "-H"}]


def _dictionary() -> GlobalCandidateDictionary:
    """Return an orthogonal four-candidate fixture dictionary."""
    return GlobalCandidateDictionary(
        matrix=torch.eye(4),
        candidate_ions=tuple({"candidate_ion_key": str(index)} for index in range(4)),
        mass_axis=torch.arange(4, dtype=torch.float32),
    )


def test_catalogue_condition_counts_preserves_explicit_filters() -> None:
    """The condition table records counts and the METASPACE condition dimensions."""
    frame = catalogue_condition_counts(
        FixtureCatalog(),
        (
            {"name": "positive", "filters": {"polarity": "Positive", "mz_min": 200.0, "mz_max": 900.0}},
            {"name": "negative", "filters": {"polarity": "Negative", "mz_min": 200.0, "mz_max": 900.0}},
        ),
    )

    assert frame["candidate_ion_count"].tolist() == [2, 1]
    assert frame["compound_count"].tolist() == [2, 1]
    assert frame["polarity"].tolist() == ["Positive", "Negative"]


def test_convergence_and_gradcheck_are_numerically_well_formed() -> None:
    """The notebook computations expose convergent observations and valid gradients."""
    dictionary = _dictionary()
    frame = projected_gradient_convergence(
        dictionary,
        batch_size=3,
        seed=11,
        iteration_counts=(1, 2, 8),
        synthetic_config=SyntheticDeconvolutionConfig(min_components=1, max_components=2),
    )

    assert frame.shape[0] == 9
    assert frame.groupby("iterations")["objective"].mean().iloc[-1] <= frame.groupby("iterations")["objective"].mean().iloc[0]
    assert projected_gradient_gradcheck(dictionary)


def test_identifiability_experiment_reports_rank_and_recovery_per_subset() -> None:
    """Small candidate subsets retain per-repeat dictionary diagnostics."""
    frame = identifiability_experiment(
        _dictionary(),
        candidate_counts=(2, 4),
        repeats=2,
        batch_size=3,
        seed=7,
        synthetic_config=SyntheticDeconvolutionConfig(min_components=1, max_components=1),
        solver_iterations=8,
    )

    assert frame.shape[0] == 4
    assert frame["dictionary_rank"].tolist() == [2, 2, 4, 4]
    assert (frame["support_exact_fraction"] == 1.0).all()


def test_sample_global_subdictionary_is_seeded_and_preserves_metadata() -> None:
    """A small baseline receives a reproducible subset of global candidate columns."""
    dictionary = GlobalCandidateDictionary(
        matrix=torch.eye(5),
        candidate_ions=tuple({"candidate_ion_key": str(index)} for index in range(5)),
        mass_axis=torch.arange(5, dtype=torch.float32),
    )

    first = sample_global_subdictionary(dictionary, candidate_count=3, seed=12)
    second = sample_global_subdictionary(dictionary, candidate_count=3, seed=12)

    assert first.candidate_ions == second.candidate_ions
    torch.testing.assert_close(first.matrix, second.matrix)


def test_catalogue_precompute_paths_and_command_are_reproducible(tmp_path) -> None:
    """Notebook loading and detached preprocessing share one catalogue location."""
    settings = {
        "catalogue": {
            "workspace_relative_path": "part_0_1_results/catalogue_workspace",
            "dataset_id": "candidate-universe",
        }
    }

    path = catalogue_path(settings, notebook_dir=tmp_path)

    assert path == tmp_path / "part_0_1_results/catalogue_workspace/datasets/candidate-universe/database_annotations/candidates.sqlite"
    assert "catalogue_precompute --settings settings.yaml" in run_command("settings.yaml")
