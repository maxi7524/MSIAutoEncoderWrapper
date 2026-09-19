"""Tests for deterministic synthetic global-dictionary mixtures."""

from __future__ import annotations

import torch

from msi_autoencoder_wrapper.models.architectures.types.deconvolution import (
    GlobalCandidateDictionary,
    SyntheticDeconvolutionConfig,
    SyntheticDeconvolutionGenerator,
)


def _dictionary() -> GlobalCandidateDictionary:
    """Return an orthogonal compact dictionary for analytical assertions."""
    return GlobalCandidateDictionary(
        matrix=torch.eye(4),  # (M=4, C=4)
        candidate_ions=tuple({"candidate_ion_key": str(index)} for index in range(4)),
        mass_axis=torch.arange(4, dtype=torch.float32),
    )


def test_generator_is_reproducible_and_preserves_exact_targets() -> None:
    """Same seed and epoch yield identical batches without global RNG mutation."""
    config = SyntheticDeconvolutionConfig(min_components=2, max_components=2)
    first = SyntheticDeconvolutionGenerator(_dictionary(), config, seed=17).generate(5, epoch=3)
    second = SyntheticDeconvolutionGenerator(_dictionary(), config, seed=17).generate(5, epoch=3)

    torch.testing.assert_close(first.spectra, second.spectra, rtol=0, atol=0)
    torch.testing.assert_close(first.abundances, second.abundances, rtol=0, atol=0)
    assert torch.equal(first.presence, first.abundances > 0)
    assert torch.equal(first.spectra, first.abundances)
    assert torch.equal(first.presence.sum(dim=1), torch.full((5,), 2))


def test_tic_normalization_scales_spectrum_and_abundance_together() -> None:
    """Dictionary reconstruction remains exact after normalized generation."""
    generator = SyntheticDeconvolutionGenerator(
        _dictionary(),
        SyntheticDeconvolutionConfig(min_components=1, max_components=3, normalization="tic"),
        seed=5,
    )
    batch = generator.generate(8)

    torch.testing.assert_close(batch.spectra.sum(dim=1), torch.ones(8))
    torch.testing.assert_close(
        batch.spectra,
        batch.abundances @ batch.dictionary.matrix.transpose(0, 1),
    )
