"""Tests for chemically motivated finite perturbations of TIC spectra."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msi_autoencoder_wrapper.analysis.autoencoder.latent.perturbations import (
    DETERMINISTIC_PERTURBATIONS,
    PERTURBATION_NAMES,
    PerturbationSettings,
    gaussian_kernel,
    interpolate_perturbation,
    normalize_tic,
    perturb_spectra,
    perturbation_diagnostics_frame,
)


def _spectra(rows: int = 4, bins: int = 32, seed: int = 0) -> torch.Tensor:
    """Sparse, peak-like, TIC-normalized spectra."""
    generator = torch.Generator().manual_seed(seed)
    values = torch.zeros((rows, bins))
    for row in range(rows):
        peaks = torch.randperm(bins - 4, generator=generator)[:5] + 2
        values[row, peaks] = torch.rand(5, generator=generator) + 0.1
    return values / values.sum(dim=1, keepdim=True)


class TestNormalizeTic:
    """Total-ion-current restoration."""

    def test_rows_sum_to_one(self) -> None:
        normalized = normalize_tic(torch.rand((3, 8)) + 0.1)

        torch.testing.assert_close(normalized.sum(dim=1), torch.ones(3))

    def test_empty_bins_stay_empty(self) -> None:
        spectra = torch.tensor([[0.0, 2.0, 0.0, 6.0]])

        normalized = normalize_tic(spectra)

        assert normalized[0, 0] == 0.0
        assert normalized[0, 2] == 0.0

    def test_a_spectrum_without_mass_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            normalize_tic(torch.zeros((2, 5)))


class TestGaussianKernel:
    """Convolution kernel used by the broadening perturbation."""

    def test_kernel_is_normalized_odd_and_symmetric(self) -> None:
        kernel = gaussian_kernel(1.5)

        assert kernel.numel() % 2 == 1
        assert float(kernel.sum()) == pytest.approx(1.0)
        torch.testing.assert_close(kernel, kernel.flip(0))

    @pytest.mark.parametrize("sigma", [0.0, -1.0])
    def test_non_positive_width_is_rejected(self, sigma: float) -> None:
        with pytest.raises(ValueError):
            gaussian_kernel(sigma)


class TestPerturbSpectra:
    """The four perturbation families."""

    @pytest.mark.parametrize("name", PERTURBATION_NAMES)
    def test_output_stays_on_the_simplex(self, name: str) -> None:
        generator = torch.Generator().manual_seed(1)

        result = perturb_spectra(_spectra(), name, generator=generator)

        torch.testing.assert_close(result.spectra.sum(dim=1), torch.ones(4))
        assert bool((result.spectra >= 0).all())

    def test_mz_shift_displaces_without_wrapping(self) -> None:
        spectra = torch.tensor([[0.5, 0.0, 0.0, 0.5]])

        result = perturb_spectra(spectra, "mz_shift", settings=PerturbationSettings(shift_bins=1))

        # The trailing 0.5 leaves the grid instead of reappearing at the front.
        assert float(result.spectra[0, 0]) == 0.0
        assert float(result.boundary_mass_lost[0]) == pytest.approx(0.5)
        # After renormalization the surviving peak carries all the mass.
        assert float(result.spectra[0, 1]) == pytest.approx(1.0)

    def test_mz_shift_reports_no_loss_when_the_edge_is_empty(self) -> None:
        spectra = torch.tensor([[0.4, 0.6, 0.0, 0.0]])

        result = perturb_spectra(spectra, "mz_shift", settings=PerturbationSettings(shift_bins=1))

        assert float(result.boundary_mass_lost[0]) == pytest.approx(0.0)

    def test_width_jitter_spreads_mass_into_neighbouring_bins(self) -> None:
        spectra = torch.zeros((1, 21))
        spectra[0, 10] = 1.0

        result = perturb_spectra(spectra, "width_jitter")

        assert float(result.spectra[0, 10]) < 1.0
        assert float(result.spectra[0, 9]) > 0.0
        assert float(result.spectra[0, 11]) > 0.0

    def test_deterministic_families_ignore_the_generator(self) -> None:
        spectra = _spectra()
        for name in DETERMINISTIC_PERTURBATIONS:
            first = perturb_spectra(spectra, name, generator=torch.Generator().manual_seed(1))
            second = perturb_spectra(spectra, name, generator=torch.Generator().manual_seed(999))

            torch.testing.assert_close(first.spectra, second.spectra)

    @pytest.mark.parametrize("name", sorted(set(PERTURBATION_NAMES) - DETERMINISTIC_PERTURBATIONS))
    def test_stochastic_families_follow_their_generator(self, name: str) -> None:
        spectra = _spectra()

        repeated = perturb_spectra(spectra, name, generator=torch.Generator().manual_seed(7))
        same_seed = perturb_spectra(spectra, name, generator=torch.Generator().manual_seed(7))
        other_seed = perturb_spectra(spectra, name, generator=torch.Generator().manual_seed(8))

        torch.testing.assert_close(repeated.spectra, same_seed.spectra)
        assert not torch.allclose(repeated.spectra, other_seed.spectra)

    def test_additive_noise_never_produces_negative_intensity(self) -> None:
        # A large noise fraction is the case where clipping actually binds.
        settings = PerturbationSettings(noise_fraction=5.0)

        result = perturb_spectra(
            _spectra(), "additive_noise", settings=settings,
            generator=torch.Generator().manual_seed(3),
        )

        assert bool((result.spectra >= 0).all())

    def test_intensity_noise_introduces_no_mass_into_empty_bins(self) -> None:
        # Multiplicative noise cannot create a peak where there was none, unlike the
        # additive family; the diagnostic must reflect that difference.
        result = perturb_spectra(
            _spectra(), "intensity_lognormal", generator=torch.Generator().manual_seed(4)
        )

        assert float(result.introduced_mass.max()) == pytest.approx(0.0)

    def test_shift_introduces_mass_where_bins_were_empty(self) -> None:
        result = perturb_spectra(_spectra(), "mz_shift")

        assert float(result.introduced_mass.max()) > 0.0

    def test_unknown_family_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            perturb_spectra(_spectra(), "not_a_perturbation")


class TestInterpolatePerturbation:
    """Amplitude scaling of a fixed perturbation direction."""

    def test_zero_amplitude_returns_the_original(self) -> None:
        spectra = _spectra()
        perturbed = perturb_spectra(spectra, "mz_shift").spectra

        blended = interpolate_perturbation(spectra, perturbed, 0.0)

        torch.testing.assert_close(blended, spectra)

    def test_unit_amplitude_returns_the_perturbation(self) -> None:
        spectra = _spectra()
        perturbed = perturb_spectra(spectra, "mz_shift").spectra

        blended = interpolate_perturbation(spectra, perturbed, 1.0)

        torch.testing.assert_close(blended, perturbed)

    def test_intermediate_amplitudes_stay_on_the_simplex(self) -> None:
        spectra = _spectra()
        perturbed = perturb_spectra(spectra, "width_jitter").spectra

        for amplitude in (0.1, 0.5, 0.9):
            blended = interpolate_perturbation(spectra, perturbed, amplitude)

            torch.testing.assert_close(blended.sum(dim=1), torch.ones(spectra.shape[0]))
            assert bool((blended >= 0).all())

    def test_displacement_grows_monotonically_with_amplitude(self) -> None:
        spectra = _spectra()
        perturbed = perturb_spectra(spectra, "mz_shift").spectra

        distances = [
            float(torch.linalg.vector_norm(interpolate_perturbation(spectra, perturbed, a) - spectra, dim=1).mean())
            for a in (0.0, 0.25, 0.5, 1.0)
        ]

        assert distances == sorted(distances)


class TestDiagnosticsFrame:
    """Per-family validity summary."""

    def test_every_family_is_reported_with_its_determinism_flag(self) -> None:
        generator = torch.Generator().manual_seed(11)
        results = {
            name: perturb_spectra(_spectra(), name, generator=generator)
            for name in PERTURBATION_NAMES
        }

        records = perturbation_diagnostics_frame(results)

        assert {record["perturbation"] for record in records} == set(PERTURBATION_NAMES)
        flags = {record["perturbation"]: record["deterministic"] for record in records}
        assert flags["mz_shift"] and flags["width_jitter"]
        assert not flags["additive_noise"] and not flags["intensity_lognormal"]

    def test_displacement_is_reported_in_input_space(self) -> None:
        results = {"mz_shift": perturb_spectra(_spectra(), "mz_shift")}

        record = perturbation_diagnostics_frame(results)[0]

        assert record["mean_input_displacement"] > 0.0
        assert record["max_input_displacement"] >= record["mean_input_displacement"]
